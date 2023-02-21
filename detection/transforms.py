"""
Custom transforms to be applied on image and mask samples.
"""
import torch
import numpy as np
from torchvision.transforms import Normalize


class ToTensorImage:
    """Gives input image array the proper format before applying
    the model."""
    def __call__(self, image):
        image = image / np.max(image)
        image = np.expand_dims(image, axis=0)
        tensor = torch.as_tensor(
            image,
            dtype=torch.float
        )
        return tensor


class ToTensorMask:
    """Gives input mask array the proper format before applying
    the loss function."""
    def __init__(self, foreground_label):
        self.foreground_label = foreground_label

    def __call__(self, mask):
        mask = (mask / self.foreground_label).astype('uint8')
        tensor = torch.as_tensor(
            mask,
            dtype=torch.long
        )
        return tensor


class Standardize:
    """Apply Z-score normalization to a given input tensor,
    i.e. re-scaling the values to be 0-mean and 1-std."""
    def __call__(self, tensor):
        norm = Normalize(torch.mean(tensor), torch.std(tensor))
        return norm(tensor)