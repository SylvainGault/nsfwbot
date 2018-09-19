import logging
import io
import multiprocessing.pool as pool
import threading
import queue
import requests
import numpy as np
import libnsfw



class AsyncWorkflow(object):
    """
    Handle the whole workflow from downloading an URL to preprocessing and
    analyzing the image. Everything is done asynchronously using callbacks.
    """

    def __init__(self, maxdownloads=3, maxdlsize=None):
        self._model = libnsfw.NSFWModel()
        self._maxdlsize = maxdlsize
        self._dlpool = pool.ThreadPool(maxdownloads)
        self._ppth = threading.Thread(target=self._preprocess)
        self._evalth = threading.Thread(target=self._evalframes)
        self._filesq = queue.Queue(10)
        self._framesq = queue.Queue(20)

        self._ppth.start()
        self._evalth.start()



    def addurl(self, url, callback=None, error_callback=None):
        def errcb(exc):
            if error_callback:
                error_callback(exc)
            else:
                raise exc

        args = (url, callback)
        self._dlpool.apply_async(self._dlimage, args, error_callback=errcb)



    def stop(self):
        self._filesq.put(None)
        self._filesq.join()



    def _dlimage(self, url, cb):
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
        self._filesq.put((cb, totalsize, trunc, f))



    def _preprocess(self):
        while True:
            task = self._filesq.get()

            if task is None:
                self._framesq.put(None)
                self._framesq.join()
                self._filesq.task_done()
                break

            cb, totalsize, trunc, f = task
            _, frames = self._model.preprocess_files([f])
            self._framesq.put((cb, totalsize, trunc, frames))
            self._filesq.task_done()

        logging.debug("Preprocessing thread quits")



    def _evalframes(self):
        stop = False

        while not stop:
            # Wait for at least one task
            task = self._framesq.get()
            if task is None:
                self._framesq.task_done()
                break

            tasks = [task]

            # Process all the pending tasks at once
            while True:
                try:
                    task = self._framesq.get_nowait()
                except queue.Empty:
                    break

                if task is None:
                    stop = True
                    self._framesq.task_done()
                    break

                tasks.append(task)


            # Concatenate all the frames to process them in a single batch
            taskidx = []
            frames_list = []
            for i, task in enumerate(tasks):
                cb, totalsize, trunc, frames = task
                frames_list.append(frames)
                taskidx += [i] * frames.shape[0]

            frames = np.concatenate(frames_list, axis=0)
            scores = self._model.eval(frames)

            # Call all the callbacks
            taskidx = np.array(taskidx)
            for i, (cb, totalsize, trunc, _) in enumerate(tasks):
                score = scores[taskidx == i].max()
                cb(totalsize, trunc, score)
                self._framesq.task_done()


        logging.debug("Evaluation thread quits")
