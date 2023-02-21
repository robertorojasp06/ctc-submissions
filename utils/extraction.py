"""
Functions required to extract/insert patches from/to volumes
using specified centers.
"""
import numpy as np
import os
import copy
from skimage import io


def extract_patches(volume, centers, size, path_to_output=None,
                    extension='.tif', squeeze=False):
    """
    Returns a list of patches from input volume according to the
    specified centers.

    Parameters
    ----------
    volume :  array
        Input volume.
    centers : list
        List of positions of patch centers.
        Each center is an array with coordinates expresssed as
        [slices, rows, columns].
    size : tuple
        Size of patches expressed as (slices, rows, columns).
    path_to_output : str, optional
        Path to the folder to save output image files.
        If set, no output list is returned.
    extension : str, optional
        File format for the output images.
    squeeze : bool, optional
        If true, patches are squeezed.

    Returns
    -------
    patches : list
        List of patches (arrays).
    """
    if not isinstance(centers, list):
        centers = [centers]
    patches = []
    shape = (volume.shape[0]+2*size[0],
             volume.shape[1]+2*size[1],
             volume.shape[2]+2*size[2])
    padded = np.zeros(shape, dtype=volume.dtype)
    padded[size[0]:(size[0]+volume.shape[0]),
           size[1]:(size[1]+volume.shape[1]),
           size[2]:(size[2]+volume.shape[2])] = volume
    for idx, center in enumerate(centers):
        starting_z = center[0]+size[0]-int(size[0]/2.0)
        starting_row = center[1]+size[1]-int(size[1]/2.0)
        starting_col = center[2]+size[2]-int(size[2]/2.0) 
        patch = padded[starting_z:(starting_z+size[0]),
                       starting_row:(starting_row+size[1]),
                       starting_col:(starting_col+size[2])]
        if squeeze:
            patch = patch.squeeze()
        if path_to_output:
            io.imsave(
                os.path.join(path_to_output, f'patch_{idx+1}{extension}'),
                patch.astype(volume.dtype),
                check_contrast=False
            )
        else:
            patches.append(patch)
    if not path_to_output:
        return patches[0] if len(centers) == 1 else patches


def insert_patch(volume, patch, center, type='positive'):
    """
    Return a volume with the specified patch inserted in the input volume.

    Parameters
    ----------
    volume : array
        Input volume to be filled.
    patch : array
        Patch to be inserted into the volume.
    type : str, optional
        Strategy to insert the patch into the volume. Allowed: 'all' (all
        patch voxels are inserted), 'positive' (only positive patch voxels
        are inserted).

    Returns
    -------
    output_image : array
        Output image with inserted patches.
    """
    output_image = copy.deepcopy(volume)
    starting_z = center[0]-int(patch.shape[0]/2.0)
    starting_row = center[1]-int(patch.shape[1]/2.0)
    starting_col = center[2]-int(patch.shape[2]/2.0)
    if type == 'all':
        output_image[starting_z:(starting_z+patch.shape[0]),
                     starting_row:(starting_row+patch.shape[1]),
                     starting_col:(starting_col+patch.shape[2])] = patch
    elif type == 'positive':
        slices, rows, cols = np.where(patch > 0)
        for idx in range(len(slices)):
            output_image[starting_z+slices[idx],
                         starting_row+rows[idx],
                         starting_col+cols[idx]] = patch[slices[idx],
                                                         rows[idx],
                                                         cols[idx]]
    else:
        raise ValueError("parameter 'type' not supported.")
    return output_image
