import cPickle
import gzip
import os
from os import path
import sys
import getopt

import numpy

from PIL import Image


def parse_args(argv):
    inputfile = ''
    out_dir = ''
    help_msg = 'unpack_mnist.py -i <InputMnistFile> -o <OutputDir>'
    try:
        opts, args = getopt.getopt(argv,"hi:o:")
    except getopt.GetoptError:
        print help_msg
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            out_dir = arg
    return inputfile, out_dir


def load_data(dataset):
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set


def save_data_to_dir(data, out_dir):
    x, y = data
    class_count = numpy.max(y) + 1
    class_index = numpy.zeros(class_count, dtype='int32')
    for i in xrange(x.shape[0]):
        example = x[i]
        example *= 255
        example = example.reshape(28, 28)
        img = Image.fromarray(example);
        if img.mode != 'L':
            img = img.convert('L')
        c = y[i]
        path = os.path.join(out_dir, str(c))
        if not os.path.exists(path):
            # Directory does not exist so we create it.
            os.makedirs(path)
        img_path = os.path.join(path, str(class_index[c]) + '.png')
        class_index[c] += 1
        img.save(img_path)


def main(argv):
    inputfile, out_dir = parse_args(argv)
    train_set, valid_set, test_set = load_data(inputfile)
    # save_data_to_dir(train_set, os.path.join(out_dir, 'train'))
    save_data_to_dir(vliad_set, os.path.join(out_dir, 'valid'))
    # save_data_to_dir(test_set, os.path.join(out_dir, 'test'))

if __name__ == "__main__":
    main(sys.argv[1:])