cd ../..

python mmaction2/tools/train.py experiments/wlasl/models/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb_2.py \
     --work-dir work_dirs/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb/0 \
     --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/wlasl/models/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb.py \
    --work-dir work_dirs/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb/1 \
    --validate --seed 1 --deterministic

python mmaction2/tools/train.py experiments/wlasl/models/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb.py \
    --work-dir work_dirs/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb/2 \
    --validate --seed 2 --deterministic
