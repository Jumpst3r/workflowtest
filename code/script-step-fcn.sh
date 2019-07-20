#!/bin/sh

# Takes a BINARIZED input image and saves the raw result of the FCN network, which is a [WxHx3] probability matrix.
# This matrix can be converted to an image with the script-step-max-rgb.sh script or can be improved with the
# script-step-crf.sh script, which applies CRF post-processing

inputImage=${1}
outputFolder=${2}

/opt/conda/etc/profile.d/conda.sh && conda activate myenv
python /input/classifier_fcnn.py ${inputImage} ${outputFolder}

