## Setting up WLASL

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

The WLASL100 can be easily downloaded and extracted by running the following:

```bash
bash setup.sh
```

### Getting other variants of WLASL

To set up for WLASL300, WLASL1000 or WLASL3000, replace the ```json``` variable inside the ```setup.sh``` file with ```"nslt_<num_classes>.json"```. Options include ```nslt_100.json```, ```nslt_300.json```, ```nslt_1000.json```, and ```nslt_2000.json```. Then, run:

```bash
bash setup.sh
```

### Getting custom variants

To get custom subsets of the dataset, e.g. WLASL10, open ```custom_setup.sh``` and change the ```n_classes``` variable value to the desired number of classes. Then, run:
```bash
bash custom_setup.sh
```

### Setting up a symbolic link
This step is optional. If you want to set up an external disk to store the dataset, open ```symbolic.sh``` and change the ```EXTERNALDRIVE``` variable from ```Sadat``` to your drive name. Then, run the script:
```
bash symbolic.sh
```
This will create set up a symbolic link called data in the current to point to the ```EXTERNALDRIVE/<dataset>/data```.
