import logging
import io
import multiprocessing.pool as pool
import threading
import queue
import requests
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
        self._evalth = threading.Thread(target=self._evalimg)
        self._filesq = queue.Queue(10)

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



    def _evalimg(self):
        while True:
            task = self._filesq.get()
            if task is None:
                self._filesq.task_done()
                break

            cb, totalsize, trunc, f = task
            _, scores = self._model.eval_files([f])
            cb(totalsize, trunc, scores)
            self._filesq.task_done()

        logging.debug("Evaluation thread quits")
