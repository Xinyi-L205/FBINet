# FBINet
## Introduction
This study focuses on predicting multi-person movements. By analyzing a sequence of historical skeletal joint positions, we aim to forecast future postural changes for individuals.
## Data
The datasets we used are all sourced from [TBIFormer](https://github.com/xiaogangpeng/tbiformer), which has made the data available online for download. Please prepare the data as follows:
```
project_folder/
├── checkpoints/
│   ├── ...
├── data/
│   ├── Mocap_UMPM
│   │   ├── train_3_75_mocap_umpm.npy
│   │   ├── test_3_75_mocap_umpm.npy
│   │   ├── test_3_75_mocap_umpm_shuffled.npy
│   ├── MuPoTs3D
│   │   ├── mupots_150_2persons.npy
│   │   ├── mupots_150_3persons.npy
│   ├── mix1_6persons.npy
│   ├── mix2_10persons.npy
├── logs/
│   ├── ...
├── models/
│   ├── ...
├── utils/
│   ├── ...
├── train.py
├── test.py
```
## Requirements
* python==3.10
* matplotlib==3.5.1
* numpy==1.22.3
* scipy==1.7.3
* torch==1.12.1
* transformers==4.18.0

## Train
`python train.py`

## Test
`python test.py`
