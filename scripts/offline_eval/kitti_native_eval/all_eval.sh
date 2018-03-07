#!/bin/bash

# set -e
# set -x

# Sort by step
folders=$(ls ./$1/ | sort -V)

for folder in $folders
do
	echo "$folder" | tee -a ./results_$1.txt
	./evaluate_object_3d_offline ~/Kitti/object/training/label_2/ $1/$folder | tee -a ./results_$1.txt
done
