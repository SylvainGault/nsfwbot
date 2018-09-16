#!/usr/bin/env python3

import sys
import libnsfw



def main():
    filenames = sys.argv[1:]
    model = libnsfw.NSFWModel()
    names, scores = model.eval_filenames(filenames)
    for f, s in zip(names, scores):
        print("%s: %f" % (f, s))

    if len(names) > 0:
        print("Average:", scores.mean())



if __name__ == '__main__':
    main()
