#!/bin/bash

################
### INPUT IMAGES
#   full data
DATA="/media/Win/Data/MSRC_images/Images/"
OUT="/media/Win/Data/data_x.bin"

#   only cows
# DATA_x="/media/Win/Data/MSRC_images/cows/img/"
# OUT_x="/media/Win/Data/data_x_cows.bin"

echo "Generating input images"
time python preprocessing/proc_input_images.py $DATA_x $OUT_x

#################
### OUTPUT IMAGES
#   full data
DATA_y="/media/Win/Data/MSRC_images/GroundTruth/"
OUT_y="/media/Win/Data/data_y.bin"

#   only cows
# DATA_y="/media/Win/Data/MSRC_images/cows/seg/"
# OUT_y="/media/Win/Data/data_y_cows.bin"

echo "Generating output images"
time python preprocessing/proc_out_images.py $DATA_y $OUT_y

################
#   split data
echo "Splitting data"
time python preprocessing/split_data.py $OUT_x $OUT_y
