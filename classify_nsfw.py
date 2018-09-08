#!/usr/bin/env python3

import sys
import numpy as np
import caffe



def eval_nsfw(model, transformer, filename):
    img = caffe.io.load_image(filename)
    img = caffe.io.resize_image(img, (256, 256, 3), 3)
    w, h = img.shape[:2]
    inh, inw = model.blobs['data'].data.shape[-2:]
    offset_x = max(w - inw, 0) // 2
    offset_y = max(h - inh, 0) // 2
    img = img[offset_y:offset_y+inh, offset_x:offset_x+inw]
    img = transformer.preprocess('data', img)
    img = img.reshape((1,) + img.shape)

    output_name = next(reversed(model.blobs))
    input_name = model.inputs[0]
    all_outputs = model.forward_all(blobs=[output_name], **{input_name: img})
    return all_outputs[output_name][0, 1]



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

    for f in filenames:
        score = eval_nsfw(nsfw_model, transformer, f)
        print("%s: %f" % (f, score))



if __name__ == '__main__':
    main()
