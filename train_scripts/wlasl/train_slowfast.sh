cd ../..

python mmaction2/tools/train.py experiments/models/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py \
    --work-dir work_dirs/slowfast_r101_8x8x1_256e_kinetics400_rgb/0 --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py \
    --work-dir work_dirs/slowfast_r101_8x8x1_256e_kinetics400_rgb/1 --validate --seed 1 --deterministic
    
python mmaction2/tools/train.py experiments/models/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py \
    --work-dir work_dirs/slowfast_r101_8x8x1_256e_kinetics400_rgb/2 --validate --seed 2 --deterministic