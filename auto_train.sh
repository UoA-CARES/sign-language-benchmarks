#!/bin/bash

# Download the dataset
mkdir -p data
cd data
mkdir -p wlasl
cd wlasl
kaggle datasets download -d risangbaskoro/wlasl-processed
mkdir -p wlasl-uncompressed
unzip wlasl-processed.zip -d ./wlasl-uncompressed
cd ../..

# Delete missing clips from the json file
python tools/wlasl/fix_missing.py nslt_100.json missing.txt wlasl_100.json data/wlasl/wlasl-uncompressed

# Split the videos
python tools/wlasl/split_videos.py wlasl_100.json data/wlasl/wlasl-uncompressed

# Extra raw frames
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/test data/wlasl/rawframes/test --ext mp4 --task rgb --level 1 --num-worker 2 --new-width 128 --new-height 171 --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/train data/wlasl/rawframes/train --ext mp4 --task rgb --level 1 --num-worker 2 --new-width 128 --new-height 171 --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/val data/wlasl/rawframes/val --ext mp4 --task rgb --level 1 --num-worker 2 --new-width 128 --new-height 171 --out-format jpg --use-opencv

# Build the labels
python tools/wlasl/build_labels.py wlasl_100.json data/wlasl

# Train the model
python mmaction2/tools/train.py c3d_wlasl.py --validate --seed 0 --deterministic --gpus 1


