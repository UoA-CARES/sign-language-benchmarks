#!/bin/bash

# Set the number of classes (chooses the top_n classes with the most samples)
n=10

# Download the dataset
mkdir -p data
cd data
mkdir -p wlasl
cd wlasl
kaggle datasets download -d risangbaskoro/wlasl-processed
mkdir -p wlasl-uncompressed
unzip wlasl-processed.zip -d ./wlasl-uncompressed
cd ../..

# Delete missing clips and create a clean annotations file
python tools/wlasl/create_json.py $n data/wlasl/wlasl-uncompressed

# Split the videos
python tools/wlasl/split_videos.py "wlasl_${n}.json" data/wlasl/wlasl-uncompressed

# Extra raw frames
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/test data/wlasl/rawframes/test --ext mp4 --task rgb --level 1 --num-worker 2 --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/train data/wlasl/rawframes/train --ext mp4 --task rgb --level 1 --num-worker 2 --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/val data/wlasl/rawframes/val --ext mp4 --task rgb --level 1 --num-worker 2 --out-format jpg --use-opencv

# Build the labels
python tools/wlasl/build_labels.py "wlasl_${n}.json" data/wlasl


