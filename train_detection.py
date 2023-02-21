import os
import torch
import argparse
import pandas as pd
import json

from unet3d.model import UNet3D
from detection.training import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="""Train the 3d-Unet model for nuclei center
        detection on patches.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_images',
        type=str,
        help="Path to the images of input patches."
    )
    parser.add_argument(
        'path_to_masks',
        type=str,
        help="Path to the masks of input patches."
    )
    parser.add_argument(
        'path_to_csv',
        type=str,
        help="""Path to the csv file containing the information
        about input patches for training."""
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the directory to save output results."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help="Learning rate."
    )
    parser.add_argument(
        '--foreground_weight',
        type=float,
        default=0.7,
        help="Foreground weight for cross entropy loss."
    )
    parser.add_argument(
        '--f_maps',
        type=int,
        default=8,
        help="""Number of features maps of the Unet in the first
        layer. The number of features maps for each layer is given
        by the geometric progression: f_maps ^ k, k=1,2,3,4"""
    )
    parser.add_argument(
        '--path_to_val_csv',
        type=str,
        default=None,
        help="""Path to the csv file containing the information
        about input patches for validation."""
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help="""Number of workers for pytorch dataloader. Set to 1
        if your machine is single core."""
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        dest='use_gpu',
        help="Add this argument to use GPU."
    )
    args = parser.parse_args()
    if args.use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = UNet3D(1, 2, is_segmentation=False, f_maps=args.f_maps)
    trainer = Trainer()
    (_,
     training_losses,
     validation_performance) = trainer.train(model,
                                             args.path_to_images,
                                             args.path_to_masks,
                                             args.path_to_csv,
                                             args.epochs,
                                             args.batch_size,
                                             args.learning_rate,
                                             1 - args.foreground_weight,
                                             args.foreground_weight,
                                             args.path_to_output,
                                             device,
                                             args.workers,
                                             args.path_to_images,
                                             args.path_to_masks,
                                             args.path_to_val_csv)
    pd.DataFrame(training_losses).to_csv(
        os.path.join(args.path_to_output, 'training_losses.csv'),
        index=False
    )
    pd.DataFrame(validation_performance).to_csv(
        os.path.join(args.path_to_output, 'validation_performance.csv'),
        index=False
    )
    with open(os.path.join(args.path_to_output, 'parameters.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)


if __name__ == '__main__':
    main()
