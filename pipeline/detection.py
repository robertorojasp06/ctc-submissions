import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.feature import blob_log

from utils.splitting import Splitter
from detection.transforms import ToTensorImage, Standardize

PATCH_SIZE = (25, 128, 128)
BLOB_DETECTION = {
    'min_sigma': 1,
    'max_sigma': 2,
    'threshold': 0.5
}


class Patch:
    def __init__(self, image, corner):
        self.image = image
        self.corner = np.array(corner).astype('int')


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
            'corner': self.patches[idx].corner
        }
        if self.image_transform:
            sample['image'] = self.image_transform(sample['image'])
        if self.norm_transform:
            sample['image'] = self.norm_transform(sample['image'])
        return sample


class Detection:
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates).astype('int')

    def __eq__(self, detection):
        return True if (self.coordinates == detection.coordinates).all() else False


class Detector:
    def __init__(self, model, patch_size=PATCH_SIZE,
                 blob_params=BLOB_DETECTION, device=torch.device('cpu')):
        self.model = model.to(device)
        self.patch_size = patch_size
        self.stride = patch_size
        self.blob_params = blob_params
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

    def detect(self, volume):
        # Split volume
        splitter = Splitter(self.patch_size, self.stride)
        patches = splitter.split(volume)
        patches = [
            Patch(patch[0], patch[1])
            for patch in patches
            if np.max(patch[0]) > 0
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
        detections = []
        self.model.eval()
        for patch in tqdm(dataloader):
            patch_image = patch['image'].to(self.device)
            output = self.model(patch_image)
            output_map = F.softmax(torch.squeeze(output), dim=0).cpu().detach().numpy()[1]
            # Detect blobs
            blobs = blob_log(
                output_map,
                self.blob_params['min_sigma'],
                self.blob_params['max_sigma'],
                num_sigma = self.blob_params['max_sigma'] - self.blob_params['min_sigma'] + 1,
                threshold=self.blob_params['threshold']
            )
            # Update coordinates relative to the volume
            corner_from_volume = patch['corner'][0].numpy()
            patch_detections = [
                Detection(corner_from_volume + np.array([blob[0], blob[1], blob[2]]).astype('int'))
                for blob in blobs
            ]
            patch_detections = [
                det
                for det in patch_detections
                if det not in detections
            ]
            detections += patch_detections
        return detections
