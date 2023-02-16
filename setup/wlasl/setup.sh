#!/bin/bash

cd ../..

# Set parameters
json="nslt_100.json"
n_workers=12

# Download the dataset
mkdir -p data
cd data
mkdir -p wlasl
cd wlasl
kaggle datasets download -d sttaseen/wlasl2000-resized
unzip wlasl2000-resized.zip -d ./
cd ../..

# Split the videos
python tools/wlasl/split_videos.py $json data/wlasl/wlasl-complete

# Extra raw frames
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-complete/test data/wlasl/rawframes/test --ext mp4 --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-complete/train data/wlasl/rawframes/train --ext mp4 --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-complete/val data/wlasl/rawframes/val --ext mp4 --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv

# Build the labels
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl


