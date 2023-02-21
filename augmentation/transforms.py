"""
Module with transforms for data augmentation on 3d images and masks.
"""
import torch
import torchio as tio
import numpy as np


class RandomSampleTransform:
    def __init__(self, **kwargs):
        self.probability = kwargs.get('probability', 0.5)
        self.channel_to_mask = kwargs.get('channel_to_mask', True)

    def transform_image(self, image, transform):
        return transform(image)

    def transform_mask(self, mask, transform):
        if self.channel_to_mask:
            transformed_mask = transform(torch.unsqueeze(mask, dim=0))
            transformed_mask = torch.squeeze(transformed_mask)
        else:
            transformed_mask = transform(mask)
        return transformed_mask.long()


class RandomFlip(RandomSampleTransform):
    def __init__(self, axes, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes

    def __call__(self, sample):
        if np.random.binomial(1, self.probability):
            transform = tio.RandomFlip(self.axes, flip_probability=1.0)
            sample['image'] = self.transform_image(sample['image'], transform)
            sample['mask'] = self.transform_mask(sample['mask'], transform)
        return sample


class RandomRotation(RandomSampleTransform):
    def __init__(self, degrees, **kwargs):
        super().__init__(**kwargs)
        self.degrees = degrees

    def __call__(self, sample):
        if np.random.binomial(1, self.probability):
            degrees = np.random.uniform(self.degrees[0], self.degrees[1])
            transform = tio.RandomAffine(
                scales=0,
                degrees=(degrees, degrees, 0, 0, 0, 0)
            )
            sample['image'] = self.transform_image(sample['image'], transform)
            sample['mask'] = self.transform_mask(sample['mask'], transform)
        return sample


class RandomNoise(RandomSampleTransform):
    def __init__(self, max_mean, max_std, **kwargs):
        super().__init__(**kwargs)
        self.max_mean = max_mean
        self.max_std = max_std

    def __call__(self, sample):
        if np.random.binomial(1, self.probability):
            transform = tio.RandomNoise(self.max_mean, self.max_std)
            sample['image'] = self.transform_image(sample['image'], transform)
        return sample


class RandomBlurring(RandomSampleTransform):
    def __init__(self, max_std, **kwargs):
        super().__init__(**kwargs)
        self.max_std = max_std

    def __call__(self, sample):
        if np.random.binomial(1, self.probability):
            transform = tio.RandomBlur(self.max_std)
            sample['image'] = self.transform_image(sample['image'], transform)
        return sample


class Identity:
    def __call__(self, sample):
        return sample
