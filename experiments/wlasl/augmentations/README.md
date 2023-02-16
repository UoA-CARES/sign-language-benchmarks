## Augmentation Ablation

Here, we ran an augmentation ablation study for different augments. The config files used for MMAction2 are under ```configs```. The bash scripts to run these tests can be found under ```train_scipts/wlasl``` in the root directory of this repository.


<div align="center">
  <div style="float:left;margin-right:10px;">
  <img src="https://user-images.githubusercontent.com/67076071/219302718-4949ec8f-c648-4295-acbb-f09eb27efa53.png"
  width=700
  height=auto
  ><br>
  </div>
</div>

### Setup for Custom Augments
For CutOut and RandAugment_T, run:
```
bash copy_augs.sh
```
This will copy the implemented augments into MMAction2 and initialise them.


### Our Results

|             |                                       |       Seed 0       |                    |       Seed 1       |                    |       Seed 2       |                    |                         |                         |
|-------------|---------------------------------------|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|-------------------------|-------------------------|
| Train Set   |                Augment                | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Mean Top-1 Accuracy (%) | Mean Top-5 Accuracy (%) |
|    Train    |                [Baseline](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/no_aug/i3d_r50_32x2x1_100e_kinetics400_noaug_wlasl100_rgb_0.py)               |        13.91       |        39.35       |        16.86       |        37.28       |        14.5        |        43.2        |          15.09          |          39.94          |
|    Train    |                 Cutout                |        13.33       |        15.38       |        12.98       |        14.89       |        13.5        |        15.48       |          13.27          |          15.25          |
|    Train    |                  [Flip](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/flip/i3d_r50_32x2x1_100e_kinetics400_flip_wlasl100_rgb_0.py)                 |        13.91       |        32.54       |        16.57       |        42.31       |        18.64       |        44.38       |          16.37          |          39.74          |
|    Train    |             [MultiscaleCrop](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/multiscalecrop/i3d_r50_32x2x1_100e_kinetics400_multiscalecrop_wlasl100_rgb_0.py)            |        40.53       |        72.19       |        38.46       |        71.89       |        34.91       |        68.05       |          37.97          |          70.71          |
|    Train    |                 [Cutmix](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/cutmix/i3d_r50_32x2x1_100e_kinetics400_cutmix_wlasl100_rgb_0.py)                |        11.54       |        27.22       |         7.1        |        23.96       |        13.02       |        29.59       |          10.55          |          26.92          |
|    Train    |                 [Mixup](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/mixup/i3d_r50_32x2x1_100e_kinetics400_mixup_wlasl100_rgb_0.py)                 |        9.17        |        25.44       |        11.83       |        27.81       |        7.69        |        23.08       |           9.56          |          25.44          |
|    Train    |                 [Augmix](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/augmix/i3d_r50_32x2x1_100e_kinetics400_augmix_wlasl100_rgb_0.py)                |        33.73       |        64.2        |        29.88       |        60.36       |        34.32       |        63.91       |          32.64          |          62.82          |
|    Train    |              [RandAugment](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/randaug/i3d_r50_32x2x1_100e_kinetics400_randaug_wlasl100_rgb_0.py)              |        34.32       |        64.2        |        31.95       |        59.76       |        29.59       |        64.5        |          31.95          |          62.82          |
|    Train    |             [RandAugment-T](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/randaug_t/i3d_r50_32x2x1_100e_kinetics400_randaugt_wlasl100_rgb_0.py)             |        41.72       |        72.19       |        36.98       |        69.23       |        47.63       |        75.15       |          42.11          |          72.19          |
|             |                                       |                    |                    |                    |                    |                    |                    |                         |                         |
|    Train    |     [RandAugment-T, MultiscaleCrop](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/randaug_t_multiscalecrop/i3d_r50_32x2x1_100e_kinetics400_randaugt_multiscalecrop_wlasl100_rgb_0.py)     |        47.93       |        75.44       |        48.52       |        73.67       |        48.22       |        74.26       |          48.22          |          74.46          |
|    Train    | [RandAugment-T, MultiscaleCrop, Augmix](https://github.com/UoA-CARES/sign-language-summer-research/tree/main/experiments/wlasl/augmentations/configs/i3d/randaug_t_multiscalecrop_augmix) |        50.3        |        75.74       |        49.11       |        76.04       |        47.93       |        75.44       |          49.11          |          75.74          |
|             |                                       |                    |                    |                    |                    |                    |                    |                         |                         |
|  Train+Val  |          [MultiscaleCrop, Flip](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/stock/i3d_r50_32x2x1_100e_kinetics400_stock_wlasl100_rgb_0.py)         |        50.78       |        78.68       |        48.84       |        76.36       |        51.16       |        79.46       |          50.26          |          78.17          |
|  Train+Val  | [RandAugment-T, MultiscaleCrop, Augmix](https://github.com/UoA-CARES/sign-language-summer-research/blob/main/experiments/wlasl/augmentations/configs/i3d/rat_ms_am_val/i3d_r50_32x2x1_100e_kinetics400_randaugt_multiscalecrop_augmix_wlasl100_rgb_0.py) |        60.08       |        85.27       |        58.91       |        82.17       |        58.91       |        82.95       |           **59.3**          |          **83.46**          |
