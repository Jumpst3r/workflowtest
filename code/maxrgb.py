'''

This file takes an raw probability matrix of dimensions [W x H x 3] and applies the
argmax operation to get an images an RGB images visualizing the classes

'''

import argparse
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from fcn_helper_function import weighted_categorical_crossentropy, IoU
from img_utils import getbinim, max_rgb_filter, get_IoU, getBinclassImg, mask2rgb, rgb2mask
from keras.engine.saving import load_model
from post import crf
from skimage import img_as_float
from skimage.color import gray2rgb
import tensorflow as tf

if not sys.warnoptions:
    warnings.simplefilter("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prob_matrix", help="path to probability matrix")
    parser.add_argument("output_folder", help="output folder")
    args = parser.parse_args()
    output_folder = args.output_folder

    matrix = pickle.load(open(args.prob_matrix, "rb" ))
    io.imsave(output_folder + 'result.png', max_rgb_filter(matrix))
    print('saved result.png')
