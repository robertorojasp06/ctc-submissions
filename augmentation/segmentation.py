"""
Module with transform pipeline to apply 'on-the-fly' data augmentation
for segmentation.
"""
import numpy as np
from torchvision.transforms import RandomChoice

from augmentation import transforms

PROBABILITIES = {
    'keep_original': 0.5,
    'space': {
        'identity': 1 / 3,
        'flipping': 1 / 3,
        'rotation': 1 / 3
    },
    'intensity': {
        'gaussian_noise': 0.25,
        'blurring': 0.25,
        'brightness': 0.25,
        'slice_brightness': 0.25
    }
}


class OnlineAugmentation:
    def __init__(self, probabilities=PROBABILITIES, max_noise_mean=0,
                 max_noise_std=0.05, max_blur_std=2, min_brightness=0.7,
                 max_brightness=1.25, min_slice_brightness=0.5,
                 max_slice_brightness=0.8):
        self.probabilities = probabilities
        self.max_noise_mean = max_noise_mean
        self.max_noise_std = max_noise_std
        self.max_blur_std = max_blur_std
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_slice_brightness = min_slice_brightness
        self.max_slice_brightness = max_slice_brightness
        self.slice_brightness_min_slices = 1
        self.slice_brightness_max_slices = 2
        self.axes_options = [0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        self.angle_options = [90, 180, 270]

    def __call__(self, sample):
        if not np.random.binomial(1, self.probabilities['keep_original']):
            # Define transformations
            identity = transforms.Identity()
            flipping = transforms.RandomFlip(
                self.axes_options[np.random.randint(len(self.axes_options))],
                probability=1.0
            )
            angle_idx = np.random.randint(len(self.angle_options))
            rotation = transforms.RandomRotation(
                (self.angle_options[angle_idx], self.angle_options[angle_idx]),
                probability=1.0
            )
            noise = transforms.RandomNoise(
                self.max_noise_mean,
                self.max_noise_std,
                probability=1.0
            )
            blurring = transforms.RandomBlurring(
                self.max_blur_std,
                probability=1.0
            )
            brightness = transforms.RandomBrightness(
                self.min_brightness,
                self.max_brightness,
                probability=1.0
            )
            slice_brightness = transforms.RandomSliceBrightness(
                self.min_slice_brightness,
                self.max_slice_brightness,
                self.slice_brightness_min_slices,
                self.slice_brightness_max_slices,
                probability=1.0
            )
            spatial_transform = RandomChoice(
                [
                    identity,
                    flipping,
                    rotation
                ],
                [
                    self.probabilities['space']['identity'],
                    self.probabilities['space']['flipping'],
                    self.probabilities['space']['rotation']
                ]
            )
            intensity_transform = RandomChoice(
                [
                    noise,
                    blurring,
                    brightness,
                    slice_brightness
                ],
                [
                    self.probabilities['intensity']['gaussian_noise'],
                    self.probabilities['intensity']['blurring'],
                    self.probabilities['intensity']['brightness'],
                    self.probabilities['intensity']['slice_brightness']
                ]
            )
            sample = intensity_transform(spatial_transform(sample))
        return sample
