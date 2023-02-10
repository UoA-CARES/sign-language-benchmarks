cd ..

python mmaction2/tools/train.py mmaction2/configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py --validate --deterministic --seed 0

python mmaction2/tools/train.py mmaction2/configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py --validate --deterministic --seed 1
    
python mmaction2/tools/train.py mmaction2/configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py --validate --deterministic --seed 2