#!/bin/bash
cd data
mkdir -p wlasl
cd wlasl
kaggle datasets download -d risangbaskoro/wlasl-processed
mkdir -p wlasl-uncompressed
unzip wlasl-processed.zip -d ./wlasl-uncompressed
cd ../../tools/wlasl
python missing_fix.py nslt_1000.json missing.txt wlasl_1000.json
echo "Copying files to label-bob..."
mv -v ./wlasl-uncompressed/videos/. ../../label-bob/downloads

# Create segment labels using the notebook
# Copy the labels to label-bob
# Run extract frames
# Create separate labels for the files (train/test/val)
# Upload the dataset onto kaggle


