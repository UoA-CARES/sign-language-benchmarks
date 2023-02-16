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

json="nslt_100_1.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_1.py --deterministic --validate --seed 0

json="nslt_100_2.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_2.py --deterministic --validate --seed 0

json="nslt_100_3.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_3.py --deterministic --validate --seed 0

json="nslt_100_4.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_4.py --deterministic --validate --seed 0

json="nslt_100_5.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_5.py --deterministic --validate --seed 0

json="nslt_100_6.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_6.py --deterministic --validate --seed 0

json="nslt_100_7.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_7.py --deterministic --validate --seed 0

json="nslt_100_8.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_8.py --deterministic --validate --seed 0

json="nslt_100_9.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_9.py --deterministic --validate --seed 0

json="nslt_100_10.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_10.py --deterministic --validate --seed 0

json="nslt_100_11.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_11.py --deterministic --validate --seed 0

json="nslt_100_12.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_12.py --deterministic --validate --seed 0

json="nslt_100_13.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_13.py --deterministic --validate --seed 0

json="nslt_100_14.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_14.py --deterministic --validate --seed 0

json="nslt_100_15.json"
python tools/wlasl/build_labels.py "data/wlasl/wlasl-complete/${json}" data/wlasl
python mmaction2/tools/train.py experiments/sample_size/i3d/i3d_r50_32x2x1_50e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_15.py --deterministic --validate --seed 0


