# ctc-submissions
Submissions for Cell Tracking Challenge (CTC).

Implementation of a pipeline to produce segmentations masks for 3D video sequences
from the Fluo-N3DL-DRO CTC dataset. This dataset provides 2 video sequences each with
50 time-indexed volumes of a developing Drosophila Melanogaster embryo.

The processing pipeline relies in the use of two 3D-Unet models. The first is trained to produce
a probability map indicating the position of nuclei centers. Detected centers are obtained
by finding local maxima and a small centered patch is extracted for each detection.
The second model is trained to produce a probability map indicating the segmentation mask
of the central nucleus for each patch.

## Setup
Before running any script, you have to setup this repository in your local machine:
1. Open a terminal inside the local repository folder.
2. Create the conda environment `ctc-second-submission` by running:
```
./create_env.sh
```

## How to run executables to reproduce submitted results
1. Download the trained models from [this link](https://drive.google.com/file/d/1LMNqzilxm0us4UULOT3LHTkkybe5-uTz/view?usp=share_link).
2. Unzip the downloaded file and move the extracted folder `models` to the local repository folder.
3. Run bash scripts:
```
./Fluo-N3DL-DRO-01.sh
```
```
./Fluo-N3DL-DRO-02.sh
```

## How to train models
1. Download the training data from [this link](https://drive.google.com/file/d/19PR8EMcDpdp3fxlh6Bgag-T5k4-AijXj/view?usp=sharing). Access to data is currently restricted. Please contact Roberto Rojas (<mailto:roberto.rojasp06@gmail.com>) to request permission for downloading the data.
2. Unzip the downloaded file and move the extracted folder `data` to the local repository folder.
3. Activate conda environment `ctc-second-submission`.
4. Train detection model:
```
python train_detection.py 'data/detection/images/' 'data/detection/masks/' 'data/detection/training_patches.csv' 'models/detection/' --learning_rate 0.01 --foreground_weight 0.75 --epochs 50 --path_to_val_csv 'data/detection/evaluation_patches.csv' --f_maps 16 --use_gpu 
```
5. Train segmentation model:
```
python train_segmentation.py 'data/segmentation/images/' 'data/segmentation/masks/' 'data/segmentation/training_patches.csv' 'models/segmentation/' --learning_rate 0.01 --foreground_weight 0.7 --epochs 100 --path_to_val_csv 'data/segmentation/evaluation_patches.csv' --f_maps 8 --use_gpu
```

## How to predict segmentation masks
1. Download the trained models from [this link](https://drive.google.com/file/d/1LMNqzilxm0us4UULOT3LHTkkybe5-uTz/view?usp=share_link).
2. Unzip the downloaded file and move the extracted folder `models` to the local repository folder.
3. Predict segmentation masks for input volumes stored in <path/to/volumes>:
```
python predict.py <path/to/volumes> <path/to/output> --use_gpu
```
