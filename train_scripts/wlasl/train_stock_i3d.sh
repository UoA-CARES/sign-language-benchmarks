cd ../..

python mmaction2/tools/train.py experiments/wlasl/augmentations/configs/i3d/stock/i3d_r50_32x2x1_100e_kinetics400_stock_wlasl100_rgb_0.py --deterministic --seed 0 --validate

python mmaction2/tools/train.py experiments/wlasl/augmentations/configs/i3d/stock/i3d_r50_32x2x1_100e_kinetics400_stock_wlasl100_rgb_0.py --deterministic --seed 1 --validate
    
python mmaction2/tools/train.py experiments/wlasl/augmentations/configs/i3d/stock/i3d_r50_32x2x1_100e_kinetics400_stock_wlasl100_rgb_0.py --deterministic --seed 2 --validate
