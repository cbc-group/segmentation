"""
This module includes functions that are essential to packaging data for a specific sink.
"""
import logging

import h5py
import numpy as np
from dask import delayed

__all__ = ["pack_arrays", "pack_itk_snap"]

logger = logging.getLogger("segmentation.pipeline.steps")


@delayed
def pack_arrays(uri: str, raw, label=None, overwrite: bool = True):
    """
    Pack the array pair as an HDF5 file.

    Args:
        uri (str): destination file
        raw (array-like): the raw data
        label (array-like, optional): the label
        overwrite (bool, optional): overwrite destination file

    Returns:
        (str): the URI
    """
    if label:
        # we need to make sure they have the same shape
        assert (
            raw.shape == label.shape
        ), "shape mis-matched between raw data and the label"

    mode = "w" if overwrite else "x"
    with h5py.File(uri, mode) as h:
        h["raw"] = raw
        if label:
            if label.dtype != np.uint8:
                logger.warning("label force cast into u8")
            h["label"] = label.astype(np.uint8)

    return uri


@delayed
def _read_nifti(path):
    try:
        import SimpleITK as sitk
    except ImportError:
        logger.error('requires "SimpleITK" to load the label')
        raise

    label = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(label)


@delayed
def pack_itk_snap(dst: str, raw: str, label: str, overwrite: bool = True):
    """
    Pack raw-label product from an ITK-SNAP project.

    Args:
        dst (str): destination file
        raw (str): the source NRRD file
        label (str): the NIFTI label file
        overwrite (bool, optional): overwrite destination file
    """
    try:
        import imageio
    except ImportError:
        logger.error('requires "imageio" to load the source')
        raise

    raw = delayed(imageio.volread)(raw)  # volread can actually handle 2-D as well
    label = _read_nifti(label)

    assert raw.shape == label.shape, "shape mis-matched between raw data and the label"

    mode = "w" if overwrite else "x"
    with h5py.File(dst, mode) as h:
        h["raw"] = raw
        h["label"] = label.astype(np.uint8)
