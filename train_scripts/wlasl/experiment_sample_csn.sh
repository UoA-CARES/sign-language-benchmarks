cd ../..

# Extract rawframes
json="nslt_100.json"
n_workers=12

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
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-complete/val data/wlasl/rawframes/train --ext mp4 --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv

# Experiment with samples per class
bash create_subset.sh

# json="nslt_100_1.json"
# python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
# python mmaction2/tools/train.py experiments/sample_size/csn/1.py --deterministic --validate --seed 0

# json="nslt_100_2.json"
# python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
# python mmaction2/tools/train.py experiments/sample_size/csn/2.py --deterministic --validate --seed 0

# json="nslt_100_3.json"
# python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
# python mmaction2/tools/train.py experiments/sample_size/csn/3.py --deterministic --validate --seed 0

# json="nslt_100_4.json"
# python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
# python mmaction2/tools/train.py experiments/sample_size/csn/4.py --deterministic --validate --seed 0

# json="nslt_100_5.json"
# python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
# python mmaction2/tools/train.py experiments/sample_size/csn/5.py --deterministic --validate --seed 0

# json="nslt_100_6.json"
# python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
# python mmaction2/tools/train.py experiments/sample_size/csn/6.py --deterministic --validate --seed 0

json="nslt_100_7.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/csn/7.py --deterministic --validate --seed 0

json="nslt_100_8.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/csn/8.py --deterministic --validate --seed 0

json="nslt_100_9.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/csn/9.py --deterministic --validate --seed 0

json="nslt_100_10.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/csn/10.py --deterministic --validate --seed 0

json="nslt_100_11.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/csn/11.py --deterministic --validate --seed 0

json="nslt_100_12.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/csn/12.py --deterministic --validate --seed 0

json="nslt_100_13.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/csn/13.py --deterministic --validate --seed 0

json="nslt_100_14.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/csn/14.py --deterministic --validate --seed 0

json="nslt_100_15.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/csn/15.py --deterministic --validate --seed 0


