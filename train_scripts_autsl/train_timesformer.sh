cd ..

python mmaction2/tools/train.py experiments/autsl/models/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py \
    --work-dir work_dirs/timesformer_divST_8x32x1_15e_kinetics400_rgb/0 --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/autsl/models/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py \
    --work-dir work_dirs/timesformer_divST_8x32x1_15e_kinetics400_rgb/1 --validate --seed 1 --deterministic

python mmaction2/tools/train.py experiments/autsl/models/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py \
    --work-dir work_dirs/timesformer_divST_8x32x1_15e_kinetics400_rgb/2 --validate --seed 2 --deterministic
