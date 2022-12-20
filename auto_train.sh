conda activate wlasl
bash setup.sh
python mmaction2/tools/train.py experiments/augmentations/configs/i3d/i3d_r50_32x2x1_200e_kinetics400_base_wlasl10_rgb.py --validate --seed 0 --deterministic --gpus 1
python mmaction2/tools/train.py experiments/augmentations/configs/i3d/i3d_r50_32x2x1_800e_kinetics400_base_wlasl10_rgb.py --validate --seed 0 --deterministic --gpus 1
python mmaction2/tools/train.py experiments/augmentations/configs/i3d/i3d_r50_32x2x1_100e_kinetics400_randaug_wlasl10_rgb.py --validate --seed 0 --deterministic --gpus 1
python mmaction2/tools/train.py experiments/augmentations/configs/i3d/i3d_r50_32x2x1_100e_kinetics400_randaug_wlasl10_rgb.py --validate --seed 1 --deterministic --gpus 1
python mmaction2/tools/train.py experiments/augmentations/configs/i3d/i3d_r50_32x2x1_100e_kinetics400_randaug_wlasl10_rgb.py --validate --seed 2 --deterministic --gpus 1

