cd ../..

python mmaction2/tools/train.py experiments/models/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb.py  \
    --work-dir work_dirs/tin_r50_1x1x8_40e_sthv2_rgb/0 --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb.py  \
    --work-dir work_dirs/tin_r50_1x1x8_40e_sthv2_rgb/1 --validate --seed 1 --deterministic
    
python mmaction2/tools/train.py experiments/models/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb.py  \
    --work-dir work_dirs/tin_r50_1x1x8_40e_sthv2_rgb/2 --validate --seed 2 --deterministic