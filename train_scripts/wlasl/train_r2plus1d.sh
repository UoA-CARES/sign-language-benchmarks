cd ../..

python mmaction2/tools/train.py experiments/models/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py \
    --work-dir work_dirs/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/0 \
    --validate --seed 0 --deterministic


python mmaction2/tools/train.py experiments/models/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py \
    --work-dir work_dirs/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/1 \
    --validate --seed 1 --deterministic


python mmaction2/tools/train.py experiments/models/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py \
    --work-dir work_dirs/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/2 \
    --validate --seed 2 --deterministic

