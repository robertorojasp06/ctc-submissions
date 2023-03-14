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
1. Download the trained models from [this link](https://drive.google.com/file/d/1AWZsEGoatcrchcGLRvbZE6vIPXcihYJG/view?usp=sharing).
2. Unzip the downloaded file and move the extracted folder `models` to the local repository folder.
3. Run bash scripts:
```
./Fluo-N3DL-DRO-01.sh
```
```
./Fluo-N3DL-DRO-02.sh
```

## How to train models

## How to predict segmentation masks
