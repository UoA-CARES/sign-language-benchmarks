cd ../..

python mmaction2/tools/train.py experiments/models/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb.py \
    --work-dir work_dirs/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb/0 \
    --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb.py \
    --work-dir work_dirs/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb/1 \
    --validate --seed 1 --deterministic
    
python mmaction2/tools/train.py experiments/models/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb.py \
    --work-dir work_dirs/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb/2 \
    --validate --seed 2 --deterministic