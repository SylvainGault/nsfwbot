import logging
import io
import multiprocessing.pool as pool
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



    def addurl(self, url, callback=None, error_callback=None):
        def cb(res):
            if callback:
                callback(*res)

        def errcb(exc):
            if error_callback:
                error_callback(exc)
            else:
                raise exc

        self._dlpool.apply_async(self._dlimage, (url,), callback=cb, error_callback=errcb)



    def _dlimage(self, url):
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
        _, scores = self._model.eval_filenames([f])
        return totalsize, trunc, scores
