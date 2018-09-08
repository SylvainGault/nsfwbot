#!/usr/bin/env python3

import sys
import numpy as np
import caffe



def eval_nsfw(model, transformer, filenames):
    imgs = []
    retfilenames = []
    for filename in filenames:
        try:
            img = caffe.io.load_image(filename)
        except:
            continue

        # Use only the first frame of a gif
        if img.ndim == 4:
            img = img[0, :, :, :]

        try:
            img = transformer.preprocess('data', img)
        except:
            continue

        retfilenames.append(filename)
        imgs.append(img)

    imgs = np.array(imgs)

    output_name = next(reversed(model.blobs))
    input_name = model.inputs[0]
    all_outputs = model.forward_all(blobs=[output_name], **{input_name: imgs})
    return retfilenames, all_outputs[output_name][:, 1]



def main():
    filenames = sys.argv[1:]
    model_def_filename = "open_nsfw_model/deploy.prototxt"
    model_weights_filename = "open_nsfw_model/resnet_50_1by2_nsfw.caffemodel"

    nsfw_model = caffe.Net(model_def_filename, caffe.TEST, weights=model_weights_filename)

    transformer = caffe.io.Transformer({'data': nsfw_model.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # Channel first format
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel

    names, scores = eval_nsfw(nsfw_model, transformer, filenames)
    for f, s in zip(names, scores):
        print("%s: %f" % (f, s))

    if len(names) > 0:
        print("Average:", scores.mean())



if __name__ == '__main__':
    main()
