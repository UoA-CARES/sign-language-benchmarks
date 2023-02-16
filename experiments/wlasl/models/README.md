## Model Evaluation

Here, we tested different models' performances on WLASL100 and WLASL2000.

The bash scripts and the config files can be found under ```train_scipts/wlasl``` and ```recognition``` respectively. 

#### Rerunning the experiments

Make sure to change the wandb logger to text logger before running the configs using MMAction2: 

```python
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
    
```


#### WLASL100 Results

|             |       Seed 0       |                    |       Seed 1       |                    |       Seed 2       |                    |      Mean      |                |
|:-----------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:--------------:|:--------------:|
|    Model    | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Top-1 Acc. (%) | Top-5 Acc. (%) |
|     [C3D](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py)     |        50.77       |        79.45       |        48.45       |        79.46       |        49.22       |        79.84       |      49.48     |      79.58     |
|     [TSN](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py)     |        44.57       |        78.29       |        47.67       |        78.68       |        45.61       |        77.78       |      45.95     |      78.25     |
|     [TSM](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb.py)     |        55.81       |        86.82       |        55.81       |        87.6        |        55.81       |        86.05       |      55.81     |      86.82     |
|    [Tanet](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/tanet/tanet_r50_1x1x8_50e_sthv1_rgb.py)    |        58.91       |        82.95       |        57.36       |        87.21       |        60.08       |        84.11       |      58.78     |      84.75     |
|     [CSN](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_150e_randaug_kinetics400_rgb.py)     |        77.91       |        93.02       |        79.84       |        93.02       |        79.84       |        93.02       |      79.20     |      93.02     |
| [Timesformer](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py) |        62.4        |        84.88       |        61.63       |        84.11       |        61.24       |        84.11       |      61.76     |      84.37     |
|   [R(2+1)d](https://github.com/UoA-CARES/sign-language-summer-research/tree/main/experiments/wlasl/models/recognition/r2plus1d)   |        23.26       |        12.02       |        29.46       |        68.6        |        30.23       |        66.67       |      27.65     |      49.10     |
|     [I3D](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb.py)     |        60.77       |        60.08       |        60.85       |        60.08       |        57.93       |        57.75       |      59.85     |      59.30     |
|     [TPN](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py)     |        67.44       |        88.37       |        62.79       |        90.7        |        58.14       |        88.37       |      62.79     |      89.15     |
|     [TIN](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb.py)     |        56.98       |        85.27       |        53.1        |        82.95       |        37.21       |        72.48       |      49.10     |      80.23     |

#### WLASL2000 Results

|           |       Seed 1       |                    |       Seed 2       |                    |      Mean      |                |
|:---------:|:------------------:|:------------------:|:------------------:|:------------------:|:--------------:|:--------------:|
|   Model   | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Top-1 Acc. (%) | Top-5 Acc. (%) |
|    [CSN](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_150e_randaug_kinetics400_rgb.py)    |        46.02       |        77.25       |        44.04       |        77.28       |      45.03     |      77.27     |
|    [TPN](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/models/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py)    |        30.64       |        65.72       |        28.45       |        62.28       |      29.55     |       64       |
