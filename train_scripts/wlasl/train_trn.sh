cd ../..

python mmaction2/tools/train.py experiments/models/recognition/trn/trn_r50_1x1x8_50e_sthv2_rgb.py \
    --work-dir work_dirs/trn_r50_1x1x8_50e_sthv2_rgb/0 \
    --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/trn/trn_r50_1x1x8_50e_sthv2_rgb.py \
    --work-dir work_dirs/trn_r50_1x1x8_50e_sthv2_rgb/1 \
    --validate --seed 1 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/trn/trn_r50_1x1x8_50e_sthv2_rgb.py \
    --work-dir work_dirs/trn_r50_1x1x8_50e_sthv2_rgb/2 \
    --validate --seed 2 --deterministic

