## Setting up AUTSL

In order to download the dataset, an existing [kaggle token](https://www.kaggle.com/docs/api#:~:text=From%20the%20site%20header%2C%20click,Create%20New%20API%20Token%E2%80%9D%20button.) needs to be set up.

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

### Downloading and extracting the dataset

AUTSL can be easily downloaded and extracted by running the following:

```bash
bash setup.sh
```

### Setting up a symbolic link
This step is optional. If you want to set up an external disk to store the dataset, open ```symbolic.sh``` and change the ```EXTERNALDRIVE``` variable from ```Sadat``` to your drive name. Then, run the script:
```
bash symbolic.sh
```
This will create set up a symbolic link called data in the current to point to the ```EXTERNALDRIVE/<dataset>/data```.
