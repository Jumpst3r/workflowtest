#!/bin/sh

# Tales a probability matrix and applies the argmax operation to get an image visualizing class labels

matrix=${1}
outputFolder=${2}

/opt/conda/etc/profile.d/conda.sh && conda activate myenv
python /input/maxrgb.py ${matrix} ${outputFolder}

