'''

This file takes an raw probability matrix of dimensions [W x H x 3] and applies the CRF algorithm.
It returns an RGB image containing the labels for printed, handwritten and background.

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", help="input image file name")
    parser.add_argument("prob_matrix", help="path to probability matrix")
    parser.add_argument("output_folder", help="output folder")
    args = parser.parse_args()
    inputim = io.imread(args.input_image)
    output_folder = args.output_folder
    out = pickle.load(open(args.prob_matrix, "rb" ))
    crf_res = crf(inputim, out)
    io.imsave(output_folder + 'fcn_out_crf.png', mask2rgb(crf_res))
    print('saved fcn_out_post.png')
