#!/bin/bash

# Download the dataset
cd data
mkdir -p wlasl
cd wlasl
kaggle datasets download -d risangbaskoro/wlasl-processed
mkdir -p wlasl-uncompressed
unzip wlasl-processed.zip -d ./wlasl-uncompressed
cd ../..

# Delete missing clips from the json file
python tools/wlasl/fix_missing.py nslt_2000.json missing.txt wlasl_2000.json data/wlasl/wlasl-uncompressed

# Split the videos
python tools/wlasl/split_videos.py wlasl_2000.json data/wlasl/wlasl-uncompressed

# Build the labels
python tools/wlasl/build_labels.py wlasl_2000.json data/wlasl

# Extra raw frames
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/test data/wlasl/rawframes/test --ext mp4 --task rgb --level 1 --num-worker 2 --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/train data/wlasl/rawframes/train --ext mp4 --task rgb --level 1 --num-worker 2 --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/val data/wlasl/rawframes/val --ext mp4 --task rgb --level 1 --num-worker 2 --out-format jpg --use-opencv



# Create segment labels using the notebook
# Copy the labels to label-bob
# Run extract frames
# Create separate labels for the files (train/test/val)
# Upload the dataset onto kaggle


