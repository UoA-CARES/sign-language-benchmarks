cd ../..

n_workers=12

# Download the dataset
mkdir -p data
cd data
mkdir -p autsl
cd autsl
kaggle datasets download -d sttaseen/autsl
unzip autsl.zip -d ./
cd ../..

# Extract raw frames
python mmaction2/tools/data/build_rawframes.py data/autsl/test data/autsl/rawframes/test --ext mp4 --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/autsl/val data/autsl/rawframes/val --ext mp4 --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/autsl/train data/autsl/rawframes/train --ext mp4 --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv

# Build labels
python tools/autsl/build_labels.py data/autsl
