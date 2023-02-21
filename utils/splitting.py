"""
Module to split a volume into 3d patches.
"""
import os
import numpy as np
from itertools import product
from skimage import io


class Splitter:
    """Class to split a volume into 3d patches.
    
    Parameters
    ----------
    patch_size : tuple
        Patch size expressed as (slices, rows, columns).
    stride : tuple
        Stride between patches expressed as (slices, rows, columns).
    """
    def __init__(self, patch_size, stride):
        self.patch_size = patch_size
        self.stride = stride

    def _pad(self, image, patch_size):
        if image.ndim != 3:
            raise ValueError("Only 3D images are accepted.")
        pad_width = (
            (0, patch_size[0]),
            (0, patch_size[1]),
            (0, patch_size[2])
        )
        return np.pad(image, pad_width)

    def split(self, image, path_to_output=None, extension='.tif',
              dtype='uint16'):
        """Split the specified image into patches.

        Parameters
        ----------
        image : array
            3D image to be splitted.

        Returns
        -------
        patches : list
            List of tuples (patch, corner), where 'patch' is the
            patch array and 'corner' is a tuple containing the slice,
            row and column indices of the patch corner in the original
            image.
        """
        patches = []
        padded = self._pad(image, self.patch_size)
        grid = product(
            range(0, image.shape[0], self.stride[0]),
            range(0, image.shape[1], self.stride[1]),
            range(0, image.shape[2], self.stride[2])
        )
        for z, i, j in grid:
            patch = padded[
                        z:z+self.patch_size[0],
                        i:i+self.patch_size[1],
                        j:j+self.patch_size[2]
                    ]
            corner = (z, i, j)
            if not path_to_output:
                patches.append((patch, corner))
            else:
                io.imsave(
                    os.path.join(
                        path_to_output,
                        f'patch_slice_{z}_row_{i}_column{j}{extension}'
                    ),
                    patch.astype(dtype),
                    check_contrast=False
                )
        if not path_to_output:
            return patches