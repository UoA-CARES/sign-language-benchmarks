## Setting up multimodal WLASL

In order to set up the multimodal dataset, an existing [kaggle token](https://www.kaggle.com/docs/api#:~:text=From%20the%20site%20header%2C%20click,Create%20New%20API%20Token%E2%80%9D%20button.) needs to be set up.

The scripts will set up the data as a RawFrameDataset which has the structure as follows:

```
sign-language-summer-research
├── data
│   └── <datset_name>
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
│       └──  val_annotations.txt
└── ...
```

### Downloading and extracting already preprocessed dataset

The preprocessed WLASL100 dataset with flow, depth and pose can be easily downloaded and extracted by running the following:

```bash
bash download_preprocessed.sh
```

### Preprocessing from scratch

To set up WLASL100 from scratch, first install the required libraries for the depth, pose and flow models:

```
cd preprocess
pip install -r requirements.txt
cd ..
```

Then run:

```bash
bash scratch_preprocess.sh
```
