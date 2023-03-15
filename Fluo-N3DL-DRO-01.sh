#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ctc-second-submission
python predict.py \
'../Fluo-N3DL-DRO/01' \
'../Fluo-N3DL-DRO/01_RES' \
--path_to_det_model 'models/detection/model_ep37.pth' \
--path_to_seg_model 'models/segmentation/model_ep66.pth' \
--use_gpu