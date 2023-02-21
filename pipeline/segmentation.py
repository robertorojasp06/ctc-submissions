import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from segmentation.transforms import ToTensorImage, Standardize
from utils.extraction import extract_patches, insert_patch

PATCH_SIZE = (9, 48, 48)


class Patch:
    def __init__(self, image, center, label):
        self.image = image
        self.center = np.array(center).astype('int')
        self.label = label
        self.mask = None

    @property
    def corner(self):
        center_from_corner = np.floor(self.image.shape / 2).astype('int')
        return self.center - center_from_corner


class PatchDataset(Dataset):
    def __init__(self, patches, image_transform, norm_transform=None):
        self.patches = patches
        self.image_transform = image_transform
        self.norm_transform = norm_transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        sample = {
            'image': self.patches[idx].image,
            'center': self.patches[idx].center,
            'label': self.patches[idx].label
        }
        if self.image_transform:
            sample['image'] = self.image_transform(sample['image'])
        if self.norm_transform:
            sample['image'] = self.norm_transform(sample['image'])
        return sample


class Segmentator:
    def __init__(self, model, patch_size=PATCH_SIZE,
                 device=torch.device('cpu')):
        self.model = model.to(device)
        self.patch_size = patch_size
        self._device = device
        self.to_tensor_image = ToTensorImage()
        self.to_normalized = Standardize()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        self.model.to(value)

    def segment(self, volume, detections):
        # Extract patches centered on detections
        patches = extract_patches(volume, detections, self.patch_size)
        patches = [
            Patch(patch, center, counter + 1)
            for counter, (patch, center) in enumerate(zip(patches, detections))
            if np.max(patch) > 0
        ]
        # Run model on patches
        dataset = PatchDataset(
            patches,
            self.to_tensor_image,
            self.to_normalized
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4)
        output_volume = np.zeros(volume.shape).astype('int')
        self.model.eval()
        for patch in tqdm(dataloader):
            patch_image = patch['image'].to(self.device)
            label = patch['label'][0].item()
            center = patch['center'][0].numpy()
            output = self.model(patch_image)
            maps = F.softmax(torch.squeeze(output), dim=0)
            mask = torch.argmax(maps, dim=0).cpu().detach().numpy().astype('bool')
            patch_object = next((x for x in patches if x.label == label), None)
            if patch_object:
                patch_object.mask = mask
            output_volume = insert_patch(
                output_volume,
                (mask * label).astype('int'),
                center
            )
        return patches, output_volume
