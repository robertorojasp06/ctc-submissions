"""
Functions required for training the model.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from detection.datasets import NucleiDetectionDataset
from detection import transforms as custom_transforms
from segmentation.performance import BinaryMetrics
from detection.performance import VolumeEvaluator
from augmentation.detection import OnlineAugmentation
from utils.peaks import find_peaks

CPU_DEVICE = torch.device('cpu')
FOREGROUND_LABEL = 255
NUM_WORKERS = 4
PEAKS_DETECTION = {
    'window_size': (5, 5, 5),
    'min_val': 0.5,
    'threshold': 0.25,
    'mode': 'reflect'
}


class Trainer:
    def __init__(self):
        self.image_transform = custom_transforms.ToTensorImage()
        self.mask_transform = custom_transforms.ToTensorMask(FOREGROUND_LABEL)
        self.augmentation_transform = OnlineAugmentation()
        self.norm_transform = custom_transforms.Standardize()

    def train(self, model, path_to_images, path_to_masks, path_to_csv, epochs,
              batch_size, learning_rate, w_background, w_foreground,
              path_to_epoch_models, device=None, num_workers=NUM_WORKERS,
              path_to_val_images=None, path_to_val_masks=None,
              path_to_val_csv=None, standardization=True, augmentation=True):
        """
        Train a pytorch model for semantic segmentation from samples in the
        specified dataset. Weighted Cross Entropy is used as loss function.

        Parameters
        ----------
        model : object
            Instantiation of a pytorch model (defined as a class).
        path_to_images : str
            Path to the folder containing the sample images to be used
            for training.
        path_to_masks : str
            Path to the folder containing the sample masks to be used
            for training.
        path_to_csv : str
            Path to the csv file containing the information about images
            and masks. Expected to have the same format as the csv produced
            by the 'sequences2patches' function in the 'datasets' module.
        epochs : int
            Number of epochs for training the model.
        batch_size : int
            Number of samples for each batch.
        learning_rate : float
            Learning rate.
        w_background : float
            Weight for background in the loss function.
            Must be in the range (0,1).
        w_foreground : float
            Weight for foreground in the loss function.
            Must be in the range (0,1).
        path_to_epoch_models : str
            Path to the folder to save pytorch models trained after
            each epoch.        
        device : torch.device object, optional
            Device for storing patches (default: None).
            Set 'device' to 'None' is equivalent to set the CPU device.
            See torch.device class to set a GPU device.
        num_workers : int, optional
            Num workers for pytorch dataloader.
        path_to_val_images : list, optional
            Path to the folder containing the sample images to be used
            for measuring validation loss.
        path_to_val_masks : list, optional
            Path to the folder containing the sample masks to be used
            for measuring validation loss.
        path_to_val_csv : str, optional
            Path to the csv file containing the information of samples
            used for validation.
        standardization : bool, optional
            Set to True to apply z-normalization to sample images before
            the model.
        augmentation : bool, optional
            Set to True to apply augmentation transforms on training
            samples.

        Returns
        -------
        model : object
            Trained model.
        training_losses : dict
            Losses computed over each training batch.
        validation_performance: dict
            Performance measured on validation samples after each
            training epoch.
        """
        # Set device to cpu if 'None'
        if device is None:
            device = CPU_DEVICE
        # Move objects to device
        model = model.to(device)
        weight_tensor = torch.as_tensor([w_background, w_foreground],
                                        dtype=torch.float).to(device)
        # Optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss(weight=weight_tensor)
        # Output lists
        training_losses = {'epoch': [], 'batch': [], 'mean_loss': []}
        validation_performance = {
            'epoch': [],
            'image_filename': [],
            'mask_filename': [],
            'loss': [],
            'pixel_accuracy': [],
            'dice': [],
            'jaccard': [],
            'targets': [],
            'output_detections': [],
            'true_positives': [],
            'false_positives': [],
            'false_negatives': [],
            'true_negatives': [],
            'true_positive_rate': [],
            'false_positive_rate': []
        }
        # Load datasets and dataloaders
        std_transform = None
        aug_transform = None
        if standardization:
            std_transform = self.norm_transform
        if augmentation:
            aug_transform = self.augmentation_transform
        datasets = {
            'train': NucleiDetectionDataset(
                path_to_images,
                path_to_masks,
                path_to_csv,
                self.image_transform,
                self.mask_transform,
                aug_transform,
                std_transform
            )
        }
        dataloders = {
            'train': DataLoader(
                datasets['train'],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
        }
        if path_to_val_images:
            datasets.update({
                'val': NucleiDetectionDataset(
                    path_to_val_images,
                    path_to_val_masks,
                    path_to_val_csv,
                    self.image_transform,
                    self.mask_transform,
                    None,
                    std_transform
                )
            })
            dataloders.update({
                'val': DataLoader(
                    datasets['val'],
                    batch_size=1,
                    shuffle=False,
                    num_workers=NUM_WORKERS
                )
            })
        # Instantiate object to measure performance
        binary_metrics = BinaryMetrics()
        # Training loop
        model.train()
        for epoch in tqdm(range(epochs)):
            for idx, train_batch in enumerate(tqdm(dataloders['train'])):
                # Move data to device
                batch_images = train_batch['image'].to(device)
                batch_masks = train_batch['mask'].to(device)
                # Reset the gradients
                model.zero_grad(set_to_none=True)
                # Compute output & loss
                output = model(batch_images)
                loss = loss_function(output, batch_masks)
                print("Training loss: {}".format(loss.item()))
                training_losses['epoch'].append(epoch + 1)
                training_losses['batch'].append(idx + 1)
                training_losses['mean_loss'].append(loss.item())
                # Backpropagation: compute gradients & update weights
                loss.backward()
                optimizer.step()
            # Measure loss on validation data
            if path_to_val_images:
                model.eval()
                epoch_val_loss = []
                for val_batch in tqdm(dataloders['val']):
                    batch_images = val_batch['image'].to(device)
                    batch_masks = val_batch['mask'].to(device)
                    val_output = model(batch_images)
                    val_loss = loss_function(val_output, batch_masks).item()
                    epoch_val_loss.append(val_loss)
                    output_maps = F.softmax(torch.squeeze(val_output), dim=0)
                    output_mask = torch.argmax(output_maps, dim=0)
                    output_map_npy = output_maps.cpu().detach().numpy()[1]
                    peaks = find_peaks(
                        output_map_npy,
                        PEAKS_DETECTION['window_size'],
                        PEAKS_DETECTION['min_val'],
                        PEAKS_DETECTION['threshold'],
                        PEAKS_DETECTION['mode']
                    )
                    peaks = np.where(peaks == True)
                    peaks = list(zip(*peaks))
                    detections = [
                        np.array([peak[0], peak[1], peak[2]]).astype('int')
                        for peak in peaks
                    ]
                    targets = pd.read_csv(
                        os.path.join(
                            datasets['val'].path_to_masks,
                            val_batch['sequence'][0],
                            val_batch['volume'][0],
                            val_batch['targets_info'][0]
                        )
                    )
                    seg_metrics = binary_metrics.all(
                        output_mask.cpu().detach().numpy().astype('bool'),
                        torch.squeeze(batch_masks).cpu().numpy().astype('bool')
                    )
                    det_evaluator = VolumeEvaluator(
                        targets.to_dict(orient='list'),
                        detections
                    )
                    det_evaluator.run_evaluation()
                    true_positives = det_evaluator.performance['detections_count']
                    false_positives = len(detections) - true_positives
                    false_negatives = len(targets) - true_positives
                    true_negatives = output_map_npy.size - len(targets) - false_positives
                    try:
                        true_positive_rate = true_positives / len(targets)
                    except ZeroDivisionError:
                        true_positive_rate = np.nan
                    try:
                        false_positive_rate = false_positives / (false_positives + true_negatives)
                    except ZeroDivisionError:
                        false_positive_rate = np.nan
                    validation_performance['epoch'].append(epoch + 1)
                    validation_performance['image_filename'].append(val_batch['image_filename'][0])
                    validation_performance['mask_filename'].append(val_batch['mask_filename'][0])
                    validation_performance['loss'].append(val_loss)
                    validation_performance['pixel_accuracy'].append(seg_metrics['pixel_accuracy'])
                    validation_performance['dice'].append(seg_metrics['dice'])
                    validation_performance['jaccard'].append(seg_metrics['jaccard'])
                    validation_performance['targets'].append(len(targets))
                    validation_performance['output_detections'].append(len(detections))
                    validation_performance['true_positives'].append(true_positives)
                    validation_performance['false_positives'].append(false_positives)
                    validation_performance['false_negatives'].append(false_negatives)
                    validation_performance['true_negatives'].append(true_negatives)
                    validation_performance['true_positive_rate'].append(true_positive_rate)
                    validation_performance['false_positive_rate'].append(false_positive_rate)
                print("Validation loss: {}".format(np.mean(epoch_val_loss)))
                model.train()
            # Save partial model
            if device != CPU_DEVICE:
                model.cpu()
            torch.save(
                model.state_dict(),
                os.path.join(path_to_epoch_models, "model_ep{}.pth".format(epoch+1))
            )
            if device != CPU_DEVICE:
                model.to(device)

        return model, training_losses, validation_performance