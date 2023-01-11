cd ..

python mmaction2/tools/train.py experiments/models/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py \
    --work-dir work_dirs/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/0 --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py \
    --work-dir work_dirs/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/1 --validate --seed 1 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py \
    --work-dir work_dirs/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/2 --validate --seed 2 --deterministic

