# sign-language-summer-research

This is a repo that contains the setup for the [WLASL dataset](https://github.com/dxli94/WLASL) and [AUTSL dataset](https://paperswithcode.com/dataset/autsl) to be used with tools available with [MMAction2](https://github.com/open-mmlab/mmaction2). Clone recursively for mmaction2:
```
git clone --recurse https://github.com/UoA-CARES/sign-language-summer-research.git
```

<div align="center">
  <div style="float:left;margin-right:10px;">
  <img src="https://user-images.githubusercontent.com/67076071/208882790-d7189c45-8a45-4d63-8b19-cfe2b7081ebf.png"
  width=700
  height=auto
  ><br>
    <p style="font-size:1.5vw;">WLASL-2000</p>
  </div>
</div>

## Directory
**After** downloading and extracting the dataset, the directory should look like below:
```
sign-language-summer-research
├── data
│   ├── autsl
│   │   └── ...
│   └── wlasl
│       ├── rawframes
│       │   ├── test
│       │   │   └── 00623
│       │   │       ├── img_00001.jpg
│       │   │       └── ...
│       │   ├── train
│       │   │   └── 00623
│       │   │       ├── img_00001.jpg
│       │   │       └── ...
│       │   └── val
│       │       └── 00626
│       │           ├── img_00001.jpg
│       │           └── ...
│       ├── test_annotations.txt
│       ├── train_annotations.txt
│       ├── val_annotations.txt
│       ├── wlasl2000-resized.zip
│       └── wlasl-complete
│           └── ...
├── setup
│   └── autsl
│       └── ...
│   └── wlasl
│       ├── setup.sh
│       └── ...
├── mmaction2
│   └── ...
├── experiments
│   ├── wlasl
│   │   ├── augmentations
│   │   │    └── ...
│   │   └── ...
│   └── autsl
│       └── ...
├── README.md
├── tools
│   ├── autsl
│   │    └── ...
│   └── wlasl
│       ├── build_labels.py
│       ├── create_annotations.ipynb
│       ├── fix_missing.py
│       ├── json_fixer.ipynb
│       ├── split_videos.ipynb
│       └── split_videos.py
└── work_dirs
    └── ...
```

In-depth information about how to set up each dataset can be found in the ```README.md``` in their respective ```setup/<dataset>``` folder.

## Requirements
### Setting up a conda environment

#### Install MiniConda
The following instructions are for Linux. For other operating systems, download and install from [here](https://docs.conda.io/en/latest/miniconda.html).
```
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
 "Miniconda3.sh"
```
Install the .sh file.
```
bash Miniconda3.sh
```
Remove the installer:
```
rm Miniconda3.sh
```
#### Creating a virtual environment
Run the following commands to create a virtual environment and to activate it:
```
conda create -n mmsign python=3.8 -y
conda activate mmsign
```
Make sure to run ```conda activate mmsign``` before running any of the scripts in this repo.

### Installing Dependencies
Install PyTorch by running the following:
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
**Note:** To fully utilise cuda, make sure to have nvidia graphics drivers installed and running. To check, run ```nvidia-smi```.


Clone the repo if not done already and go inside the repo:
```
git clone --recurse https://github.com/UoA-CARES/sign-language-summer-research.git
cd sign-language-summer-research
````
To install all the other modules, navigate to the root directory of this repo after cloning and run the following:
```
pip install -r requirements.txt
```
Install mmcv:
```
pip install openmim
mim install mmcv-full
```
Assuming current directory is the root of the repository, install mmaction2 from source:
```
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .  
cd ..
```
This one is optional but to use the conda environment in Notebook, run:
```
conda install ipykernel -y
ipython kernel install --user --name=mmsign
```
## Setup
### Downloading and extracting the dataset
In order to download the dataset, an existing [kaggle token](https://www.kaggle.com/docs/api#:~:text=From%20the%20site%20header%2C%20click,Create%20New%20API%20Token%E2%80%9D%20button.) needs to be set up.
All the data-acquisition and extraction is handled by ```setup.sh```. From the ```setup/<dataset>``` directory of the repo, run:
```
bash setup.sh
```
For WLASL, subsets can be chosen by changing the name of the ```json``` variable inside ```setup.sh```. Options include ```nslt_100.json```, ```nslt_300.json```, ```nslt_1000.json```, and ```nslt_2000.json```. More details can be found [here](https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized). 
**Note:** If on any other operating system than Linux/Mac, open the bash file and run each command one by one.

This bash script will download the dataset from kaggle (Kaggle token needs to be set up for this), extract and store the dataset under the ```data``` directory.

## Setting up WandB
Log in to WandB account by running the following on terminal:
```
wandb login
```

To link the model training with WandB, copy the following code snippet and paste at the end of the config file.

```python
# Setup WandB
log_config = dict(interval=10, hooks=[
    dict(type='TextLoggerHook'),
    dict(type='WandbLoggerHook',
         init_kwargs={
             'entity': "cares",
             'project': "wlasl-100"
         },
         log_artifact=True)
]
)
```
Since most configs are pre-setup with wandb, to undo them, replace the log_config with the following:
```python
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
```

## Training
Start training by running the following template:
```
python tools/train.py ${CONFIG_FILE} [optional arguments]
```
Example: Train a pretrained 3CD model on wlasl with periodic validation.
```
python mmaction2/tools/train.py models/c3d/c3d_16x16x1_sports1m_wlasl100_rgb.py --validate --seed 0 --deterministic --gpus 1
```

## Testing
Use the following template to test a model:
```
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```
Examples and more information can be found [here](https://github.com/open-mmlab/mmaction2/blob/master/docs/en/getting_started.md#test-a-dataset).


## Citations
```
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
```
```
@inproceedings{li2020transferring,
 title={Transferring cross-domain knowledge for video sign language recognition},
 author={Li, Dongxu and Yu, Xin and Xu, Chenchen and Petersson, Lars and Li, Hongdong},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
 pages={6205--6214},
 year={2020}
}
```
