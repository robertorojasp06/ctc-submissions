import os
import torch
import argparse
import json
from skimage import io
from glob import glob

from unet3d.model import UNet3D
from pipeline.pipeline import Pipeline

PATH_TO_DET_MODEL = os.path.join(
    'models', 'detection', 'unet3d_detection.pth'
)
PATH_TO_SEG_MODEL = os.path.join(
    'models', 'segmentation', 'unet3d_segmentation.pth'
)


def main():
    parser = argparse.ArgumentParser(
        description="""Run instance nuclei segmentation pipeline on
        a set of volumes.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_volumes',
        type=str,
        help="Path to the directory containing the input volumes."
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the directory to save output volumes."
    )
    parser.add_argument(
        '--path_to_det_model',
        type=str,
        default=PATH_TO_DET_MODEL,
        help="""Path to the weights of the trained pytorch model
        for detection."""
    )
    parser.add_argument(
        '--path_to_seg_model',
        type=str,
        default=PATH_TO_SEG_MODEL,
        help="""Path to the weights of the trained pytorch model
        for segmentation."""
    )
    parser.add_argument(
        '--path_to_config',
        type=str,
        default='predict.json',
        help="""Path to the JSON file containing the algorithm
        configuration parameters."""
    )
    parser.add_argument(
        '--extension',
        type=str,
        default='.tif',
        help="File extension of input images."
    )
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        action='store_true',
        help="Add this argument to use GPU."
    )
    args = parser.parse_args()
    if args.use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    with open(args.path_to_config, 'r') as fp:
        params = json.load(fp)
    det_model = UNet3D(
        1, 2, is_segmentation=False, f_maps=params['detection']['f_maps']
    )
    det_model.load_state_dict(torch.load(args.path_to_det_model))
    seg_model = UNet3D(
        1, 2, is_segmentation=False, f_maps=params['segmentation']['f_maps']
    )
    seg_model.load_state_dict(torch.load(args.path_to_seg_model))    
    pipeline = Pipeline(
        det_model,
        seg_model,
        det_params=params['detection'],
        seg_params=params['segmentation'],
        device=device
    )
    paths_to_volumes = sorted(glob(
        os.path.join(args.path_to_volumes, '*' + args.extension)
    ))
    for path in paths_to_volumes:
        filename = os.path.basename(path)
        volume = io.imread(path)
        _, output_volume = pipeline.run(volume)
        io.imsave(
            os.path.join(args.path_to_output, f'mask{filename[1:]}'),
            output_volume.astype('uint16'),
            check_contrast=False
        )
    params.update(vars(args))
    with open(os.path.join(args.path_to_output, 'parameters.json'), 'w') as fp:
        json.dump(params, fp, indent=4)


if __name__ == '__main__':
    main()
