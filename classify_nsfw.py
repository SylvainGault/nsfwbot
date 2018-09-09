#!/usr/bin/env python3

import sys
import numpy as np
import PIL.Image
import caffe



def load_frames(pilimg, size):
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

        frame = frame.resize(size, PIL.Image.BILINEAR)
        frame = np.array(frame).astype(np.float32) / 255.0

        retimgs.append(frame)
        frameno += 10

    return np.array(retimgs)



def eval_nsfw(model, filenames):
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



def main():
    filenames = sys.argv[1:]
    model_def_filename = "open_nsfw_model/deploy.prototxt"
    model_weights_filename = "open_nsfw_model/resnet_50_1by2_nsfw.caffemodel"

    nsfw_model = caffe.Net(model_def_filename, caffe.TEST, weights=model_weights_filename)

    names, scores = eval_nsfw(nsfw_model, filenames)
    for f, s in zip(names, scores):
        print("%s: %f" % (f, s))

    if len(names) > 0:
        print("Average:", scores.mean())



if __name__ == '__main__':
    main()
