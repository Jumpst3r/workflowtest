'''
Part of the workflow project. This script takes a binarized image as an input and returns the raw FCN result.

The input image is an array of dimensions [W x H]

The raw output of the FCN network is a matrix of dimension [W x H x 3]

The three channels correspond to the class probablitites of each pixel for the three classes, printed, handwritten and
background.

To convert this matrix to class labels, apply np.argmax
'''

import argparse
import sys
import warnings

import pickle
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


BOXWDITH = 256
STRIDE = BOXWDITH - 10

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)

def classify(image):
    try:
        model = load_model('/input/models/fcnn_bin.h5', custom_objects={
                    'loss': weighted_categorical_crossentropy([0.4,0.5,0.1]), 'IoU': IoU})
    except OSError:
        # Use relative file path if anyone tries to run the generated binary without using the provided docker image
        model = load_model('models/fcnn_bin.h5', custom_objects={
                            'loss': weighted_categorical_crossentropy([0.4,0.5,0.1]), 'IoU': IoU})
    orgim = np.copy(image)
    if len(image.shape) == 2:
        image = gray2rgb(image)
    image = img_as_float(image)
    maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
    maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
    mask = np.ones((maskh, maskw, 3))
    mask2 = np.zeros((maskh, maskw, 3))
    mask[0:image.shape[0], 0:image.shape[1]] = image
    print("classifying image...")
    for y in range(0, mask.shape[0], STRIDE):
        x = 0
        if (y + BOXWDITH > mask.shape[0]):
            break
        while (x + BOXWDITH) < mask.shape[1]:
            input = mask[y:y+BOXWDITH, x:x+BOXWDITH]
            std = input.std() if input.std() != 0 else 1
            mean = input.mean()
            mask2[y:y+BOXWDITH, x:x+BOXWDITH] = model.predict(
                    np.array([(input-mean)/std]))[0]
            x = x + STRIDE
    return mask2[0:image.shape[0], 0:image.shape[1]]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", help="input image file name")
    parser.add_argument("output_folder", help="output folder")
    args = parser.parse_args()
    inputim = io.imread(args.input_image)
    output_folder = args.output_folder

    # This is a [w x h x 3] probability matrix
    out = classify(inputim)
    crf_res = crf(inputim, out)

    # Convert to an RGB image and save
    pickle.dump(out, open(output_folder + 'fcn_raw.nparray', "wb"))
    print('saved raw prob. matrix')
