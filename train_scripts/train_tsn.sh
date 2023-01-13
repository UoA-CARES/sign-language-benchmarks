cd ..

python mmaction2/tools/train.py experiments/models/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py \
    --work-dir work_dirs/tsn_r50_1x1x3_75e_ucf101_rgb/0 \
    --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py \
    --work-dir work_dirs/tsn_r50_1x1x3_75e_ucf101_rgb/1 \
    --validate --seed 1 --deterministic
    
python mmaction2/tools/train.py experiments/models/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py \
    --work-dir work_dirs/tsn_r50_1x1x3_75e_ucf101_rgb/2 \
    --validate --seed 2 --deterministic