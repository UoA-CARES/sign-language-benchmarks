## Model Evaluation

Here, we tested the best performing WLASL models on AUTSL.

The bash scripts and the config files can be found under ```train_scipts/autsl``` and ```recognition``` respectively. 

#### Rerunning the experiments

Make sure to change the wandb logger to text logger before running the configs using MMAction2: 

```python
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
```


#### AUTSL Results

|             |       Split 1      |                    |       Split 2      |                    |       Split 3      |                    | Mean Top-1 | Mean Top-5  |
|:-----------:|:------------------:|--------------------|:------------------:|--------------------|:------------------:|--------------------|:----------:|:-----------:|
|    Model    | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Top-1 Accuracy (%) | Top-5 Accuracy (%) |  Acc. (%)  |   Acc. (%)  |
|     [CSN](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/autsl/models/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb.py)     |        90.25       |        98.72       |        93.32       |        99.36       |        89.01       |        98.72       |    90.86   |    98.93    |
| [Timesformer](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/autsl/models/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py) |        76.58       |        93.45       |        70.22       |        90.99       |        72.84       |        92.68       |    73.21   |    92.37    |
|     [TPN](https://github.com/UoA-CARES/sign-language-summer-research/tree/main/experiments/autsl/models/recognition/tpn)     |        89.82       |        98.93       |        89.23       |        98.34       |        89.54       |        98.56       |    89.53   |    98.61    |
