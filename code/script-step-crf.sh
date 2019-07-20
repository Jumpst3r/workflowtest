#!/bin/sh

# Takes the raw output of the FCN network (a probability matrix) + the original image and applies the CRF algorithm.
# returns an image visualizing the class labels

matrix=${1}
inputimage=${2}
outputFolder=${3}

/opt/conda/etc/profile.d/conda.sh && conda activate myenv
python /input/classifier_crf.py ${matrix} $(inputimage) ${outputFolder}

