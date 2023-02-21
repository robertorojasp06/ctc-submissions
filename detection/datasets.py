"""
Custom pytorch Dataset classes for detection datasets.
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io


class NucleiDetectionDataset(Dataset):
    """3d nuclei detection dataset."""
    def __init__(self, path_to_images, path_to_masks, path_to_csv,
                 image_transform=None, mask_transform=None,
                 augmentation_transform=None, norm_transform=None):
        self.path_to_images = path_to_images
        self.path_to_masks = path_to_masks
        self.patches_df = pd.read_csv(path_to_csv)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augmentation_transform = augmentation_transform
        self.norm_transform = norm_transform

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):
        image_filename = self.patches_df['image'].iloc[idx]
        mask_filename = self.patches_df['mask'].iloc[idx]
        sequence_num = self.patches_df['sequence'].iloc[idx]
        sequence = f'{sequence_num:02d}'
        volume = self.patches_df['volume'].iloc[idx]
        targets_info = self.patches_df['targets_info'].iloc[idx]
        path_to_image = os.path.join(
            self.path_to_images,
            sequence,
            volume,
            image_filename
        )
        path_to_mask = os.path.join(
            self.path_to_masks,
            sequence,
            volume,
            mask_filename
        )
        image = io.imread(path_to_image)
        mask = io.imread(path_to_mask)
        sample = {
            'image': image,
            'mask': mask,
            'image_filename': image_filename,
            'mask_filename': mask_filename,
            'volume': volume,
            'sequence': sequence,
            'targets_info': targets_info
        }
        if self.image_transform:
            sample['image'] = self.image_transform(sample['image'])
        if self.mask_transform:
            sample['mask'] = self.mask_transform(sample['mask'])
        if self.augmentation_transform:
            sample = self.augmentation_transform(sample)
        if self.norm_transform:
            sample['image'] = self.norm_transform(sample['image'])
        return sample
