cd ..

python mmaction2/tools/train.py experiments/models/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb_0.py \
    --work-dir work_dirs/0/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb \
    --validate --seed 0 --deterministic

python mmaction2/tools/train.py experiments/models/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb_1.py \
    --work-dir work_dirs/1/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb \
    --validate --seed 1 --deterministic

    python mmaction2/tools/train.py experiments/models/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb_2.py \
        --work-dir work_dirs/2/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb \
        --validate --seed 2 --deterministic
