"""
Module with the pipeline to process input volumes.

input volume -> Nuclei detection (Detector) -> list of detection objects
             -> Nuclei segmentation (Segmentator) -> list of segmented patch objects
"""
import torch

from pipeline.detection import Detector
from pipeline.segmentation import Segmentator

DET_PARAMS = {
    'patch_size': (25, 128, 128),
    'window_size': (5, 5, 5),
    'min_val': 0.5,
    'threshold': 0.25,
    'mode': 'reflect'
}
SEG_PARAMS = {
    'patch_size': (5, 48, 48)
}


class Pipeline:
    def __init__(self, det_model, seg_model, det_params=DET_PARAMS,
                 seg_params=SEG_PARAMS, device=torch.device('cpu')):
        self.det_model = det_model
        self.seg_model = seg_model
        self.det_params = det_params
        self.seg_params = seg_params
        self.device = device

    @property
    def detector(self):
        peaks_params = {
            key: DET_PARAMS[key]
            for key in ('window_size', 'min_val', 'threshold', 'mode')
        }
        return Detector(
            self.det_model,
            self.det_params['patch_size'],
            peaks_params,
            self.device
        )

    @property
    def segmentator(self):
        return Segmentator(
            self.seg_model,
            self.seg_params['patch_size'],
            self.device
        )

    def run(self, volume):
        """Run pipeline on specified volume.

        Parameters
        ----------
        volume : array
            Input 3d array to be processed.
    
        Returns
        -------
        segmented_patches : list
            List of Patch objects with segmentation masks.
        output_volume : array
            Volume with fused masks (no merging strategy).
        """
        detections = self.detector.detect(volume)
        detections = [det.coordinates for det in detections]
        return self.segmentator.segment(volume, detections)
