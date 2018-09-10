import math
import numpy as np
import PIL.Image
import caffe



def load_frames(pilimg, size):
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
        w, h = size
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

        w, h = size
        nh = math.ceil((fh - h * overlap) / (h * nonoverlap))
        nw = math.ceil((fw - w * overlap) / (w * nonoverlap))

        for hoff in np.linspace(0, fh - h, nh, dtype=np.int32):
            for woff in np.linspace(0, fw - w, nw, dtype=np.int32):
                tile = frame[hoff:hoff+h, woff:woff+w]
                retimgs.append(tile)

        frameno += 10

    return np.array(retimgs)



def eval_nsfw(filenames):
    model_def_filename = "open_nsfw_model/deploy.prototxt"
    model_weights_filename = "open_nsfw_model/resnet_50_1by2_nsfw.caffemodel"
    model = caffe.Net(model_def_filename, caffe.TEST, weights=model_weights_filename)

    transformer = caffe.io.Transformer({'data': model.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # Channel first format
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel

    size = model.blobs['data'].data.shape[2:]
    imgs = []
    retfilenames = []

    # Each filename can result in several imgs. This array hold the
    # retfilenames index for each entry in imgs.
    residx = []

    for filename in filenames:
        try:
            img = PIL.Image.open(filename)
        except:
            continue

        retfilenames.append(filename)
        frames = load_frames(img, size)

        for img in frames:
            try:
                img = transformer.preprocess('data', img)
            except:
                continue

            imgs.append(img)
            residx.append(len(retfilenames) - 1)

    imgs = np.array(imgs)
    residx = np.array(residx)

    output_name = next(reversed(model.blobs))
    input_name = model.inputs[0]
    all_outputs = model.forward_all(blobs=[output_name], **{input_name: imgs})
    all_outputs = all_outputs[output_name]
    out = [all_outputs[residx == i, 1].max() for i in range(len(retfilenames))]
    return retfilenames, np.array(out)
