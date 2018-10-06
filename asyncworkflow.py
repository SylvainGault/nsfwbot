import logging
import io
import asyncio
import concurrent.futures
import queue
import requests
import numpy as np
import libnsfw



class AsyncWorkflow(object):
    """
    Handle the whole workflow from downloading an URL to preprocessing and
    analyzing the image. Everything is done asynchronously using asyncio.
    """

    def __init__(self, loop=None, maxdownloads=3, maxdlsize=None):
        if loop is None:
            loop = asyncio.get_event_loop()

        self._model = libnsfw.NSFWModel()
        self._loop = loop
        self._maxdlsize = maxdlsize
        self._dlpool = concurrent.futures.ThreadPoolExecutor(maxdownloads)
        self._pppool = concurrent.futures.ThreadPoolExecutor(1)
        self._evalpool = concurrent.futures.ThreadPoolExecutor(1)
        self._evalq = queue.Queue()

        # Limit the number of downloaded images currently in memory
        self._semimgs = asyncio.BoundedSemaphore(10)
        self._semframes = asyncio.BoundedSemaphore(100)



    async def score_url(self, url):
        # Download image
        await self._semimgs.acquire()
        try:
            res = await self._loop.run_in_executor(self._dlpool, self._dlimg, url)
        except:
            self._semimgs.release()
            raise

        totalsize, istrunc, f = res

        # Preprocess the image file into a set of frames
        await self._semframes.acquire()
        try:
            frames = await self._loop.run_in_executor(self._pppool, self._preprocess, f)
        except:
            self._semframes.release()
            self._semimgs.release()
            raise

        del f
        self._semimgs.release()

        # Evaluate all the frames
        if frames.shape[0] > 0:
            try:
                score = await self._evalframes(frames)
            except:
                self._semframes.release()
                raise
        else:
            score = None

        del frames
        self._semframes.release()

        return totalsize, istrunc, score



    def _dlimg(self, url):
        with requests.get(url, stream=True) as r:
            totalsize = r.headers.get('Content-Length')

            # Only read up to max_download_size bytes of image.
            trunc = False
            content = bytes()
            for chunk in r.iter_content(chunk_size=1024):
                content += chunk
                if len(content) > self._maxdlsize:
                    content = content[:self._maxdlsize]
                    trunc = True
                    break

        f = io.BytesIO(content)

        if totalsize is not None:
            totalsize = int(totalsize)
        elif not trunc:
            totalsize = len(content)

        return (totalsize, trunc, f)



    def _preprocess(self, data):
        _, frames = self._model.preprocess_files([data])
        return frames



    async def _evalframes(self, frames):
        f = asyncio.Future()
        self._evalq.put((frames, f))
        self._loop.run_in_executor(self._evalpool, self._evalbatch)
        return await f



    def _evalbatch(self):
        tasks = []
        while True:
            try:
                task = self._evalq.get_nowait()
            except queue.Empty:
                break
            tasks.append(task)

        if len(tasks) == 0:
            return

        # Concatenate all the frames to process them in a single batch
        taskidx = []
        frames_list = []
        for i, task in enumerate(tasks):
            frames, fut = task
            frames_list.append(frames)
            taskidx += [i] * frames.shape[0]

        frames = np.concatenate(frames_list, axis=0)
        scores = self._model.eval(frames)

        # Set the result of all the Futures
        taskidx = np.array(taskidx)
        for i, (_, fut) in enumerate(tasks):
            score = scores[taskidx == i].max()
            fut.set_result(score)
