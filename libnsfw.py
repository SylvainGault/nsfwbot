import math
import numpy as np
import PIL.Image
import caffe



class NSFWModel(object):
    model_def_filename = "open_nsfw_model/deploy.prototxt"
    model_weights_filename = "open_nsfw_model/resnet_50_1by2_nsfw.caffemodel"

    def __init__(self, deffile=model_def_filename, weightsfile=model_weights_filename):
        model = caffe.Net(deffile, caffe.TEST, weights=weightsfile)

        # Cache some meta-informations about the model
        self.model_inshape = model.blobs['data'].data.shape[1:]
        self.model_insize = self.model_inshape[1:]
        self.model_inname = model.inputs[0]
        self.model_outname = next(reversed(model.blobs))

        transformer = caffe.io.Transformer({'data': model.blobs[self.model_inname].data.shape})
        transformer.set_transpose('data', (2, 0, 1))  # Channel first format
        transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
        transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel

        self.model = model
        self.transformer = transformer



    def _load_frames(self, pilimg):
        maxunevenresize = 0.2
        overlap = 0.5
        nonoverlap = 1 - overlap

        retimgs = []
        frameno = 0
        while True:
            try:
                pilimg.seek(frameno)
            except EOFError:
                break

            frame = pilimg
            if frame.mode != 'RGB':
                frame = frame.convert("RGB")

            fw, fh = frame.size
            w, h = self.model_insize
            if 1 - maxunevenresize <= (fw * h) / (w * fh) <= 1 + maxunevenresize:
                fh, fw = h, w
            elif fw * h > w * fh:
                fw = round(fw * h / fh)
                fh = h
            else:
                fh = round(fh * w / fw)
                fw = w

            frame = frame.resize((fw, fh), PIL.Image.BILINEAR)
            frame = np.array(frame).astype(np.float32) / 255.0

            w, h = self.model_insize
            nh = math.ceil((fh - h * overlap) / (h * nonoverlap))
            nw = math.ceil((fw - w * overlap) / (w * nonoverlap))

            for hoff in np.linspace(0, fh - h, nh, dtype=np.int32):
                for woff in np.linspace(0, fw - w, nw, dtype=np.int32):
                    tile = frame[hoff:hoff+h, woff:woff+w]
                    retimgs.append(tile)

            frameno += 10

        return np.array(retimgs)



    def eval(self, imgs):
        """Evaluate the NSFW score on some preprocessed images."""

        assert imgs.shape[0] == 0 or imgs.shape[1:] == self.model_inshape

        inname = self.model_inname
        outname = self.model_outname

        outputs = self.model.forward_all(blobs=[outname], **{inname: imgs})
        outputs = outputs[outname]

        # Empty arrays are shaped (0,) instead of (0, 2).
        outputs = outputs.reshape((-1, 2))
        return outputs[:, 1]



    def eval_pil(self, pilimgs):
        """Evaluate the NSFW score on PIL-compatible image objects."""

        imgs = []

        # Each PIL image can result in several imgs. This array hold the
        # index in pilimgs for each entry in imgs.
        residx = []

        for i, pilimg in enumerate(pilimgs):
            frames = self._load_frames(pilimg)
            for frame in frames:
                try:
                    frame = self.transformer.preprocess('data', frame)
                except:
                    continue

                imgs.append(frame)
                residx.append(i)

        imgs = np.array(imgs)
        residx = np.array(residx)

        outputs = self.eval(imgs)
        out = [outputs[residx == i].max() for i in range(len(pilimgs))]
        return np.array(out)



    def eval_filenames(self, filenames):
        """Evaluate the NSFW score on filenames."""

        imgs = []
        retfilenames = []

        for filename in filenames:
            try:
                img = PIL.Image.open(filename)
            except:
                continue

            imgs.append(img)
            retfilenames.append(filename)

        return retfilenames, self.eval_pil(imgs)
