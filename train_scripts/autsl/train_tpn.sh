cd ../..

python mmaction2/tools/train.py experiments/autsl/models/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    --work-dir work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/0 --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/autsl/models/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    --work-dir work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/1 --validate --seed 1 --deterministic

python mmaction2/tools/train.py experiments/autsl/models/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    --work-dir work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/2 --validate --seed 2 --deterministic
