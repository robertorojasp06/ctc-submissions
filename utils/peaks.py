"""
Module with function to find peaks in arrays.
"""
import numpy as np
import scipy.ndimage as ndi


def find_peaks(array, window_size, min_val=0, threshold=None,
               mode='reflect'):
    """Find local maxima in input array.

    Parameters
    ----------
    array : array
        Input numpy array.
    window_size : tuple or list
        Size of centered local window considered to find a peak in
        each position. Size for each dimension must be specified.
    min_val : float, optional
        Only peaks above min_val are returned.
    threshold : float, optional
        If specified, input array is set to zero on positions with
        values below threshold before applying maximum filter.
    mode : str, optional
        The mode parameter determines how the input array is
        extended when the filter overlaps a border. Accepted values:
        'reflect', 'constant', 'nearest', 'mirror', 'wrap'.

    Returns
    -------
    output : array
        Boolean array where True values indicate peak positions.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError('input array must be numpy array.')
    if not isinstance(window_size, (tuple, list)):
        raise ValueError('window size must be a tuple or list.')
    if array.ndim != len(window_size):
        raise ValueError("""window size must match the dimensions
                         of input array.""")
    if threshold:
        array[array < threshold] = 0
    footprint = np.ones(window_size)
    image_max = ndi.maximum_filter(
        array,
        footprint=footprint,
        mode=mode
    )
    output = array == image_max
    # no peaks for a trivial image
    image_is_trivial = np.all(output)
    if image_is_trivial:
        output[:] = False
    output &= array > min_val
    return output
