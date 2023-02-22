#!/bin/bash

cd ../../..


# Download the dataset
mkdir -p data
cd data
kaggle datasets download -d sttaseen/wlasl2000-seven-sees
unzip wlasl2000-seven-sees.zip -d ./
cd ../..



