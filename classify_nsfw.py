#!/usr/bin/env python3

import sys
import libnsfw



def main():
    filenames = sys.argv[1:]
    model = libnsfw.NSFWModel()
    idx, scores = model.eval_files(filenames)
    for i, s in zip(idx, scores):
        print("%s: %f" % (filenames[i], s))

    if len(idx) > 0:
        print("Average:", scores.mean())



if __name__ == '__main__':
    main()
