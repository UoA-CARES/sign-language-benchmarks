cd ../..

python mmaction2/tools/train.py experiments/models/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb.py  \
    --work-dir work_dirs/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb/0 \
    --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb.py  \
    --work-dir work_dirs/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb/1 \
    --validate --seed 1 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb.py  \
    --work-dir work_dirs/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb/2 \
    --validate --seed 2 --deterministic


