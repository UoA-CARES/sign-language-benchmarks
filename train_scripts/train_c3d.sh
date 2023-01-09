cd ..

python mmaction2/tools/train.py experiments/models/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
    --work-dir work_dirs/c3d_sports1m_16x1x1_45e_ucf101_rgb/0 --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
    --work-dir work_dirs/c3d_sports1m_16x1x1_45e_ucf101_rgb/1 --validate --seed 1 --deterministic
    
python mmaction2/tools/train.py experiments/models/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
    --work-dir work_dirs/c3d_sports1m_16x1x1_45e_ucf101_rgb/2 --validate --seed 2 --deterministic