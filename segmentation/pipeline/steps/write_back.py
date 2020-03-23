"""
This module includes functions that stands at the last step of the pipeline, which 
concludes the computation with a single, unified product.
"""
from dask import delayed
import logging

logger = logging.getLogger("segmentation.pipeline.steps")


def write_tiff(uri, data):
    """
    Write TIFF.

    Args:
        uri (str): target path
        data (array-like): the data
    
    Returns:
        (Future)
    """
    try:
        import imageio
    except ImportError:
        logger.error('requires "imageio"')
        raise

    if data.ndim == 2:
        func = delayed(imageio.imwrite)
    else:
        # n-dim requires volwrite (or mvolwrite?)
        func = delayed(imageio.volwrite)
    return delayed(func)(uri, data)


@delayed
def write_nifti(uri: str, data):
    try:
        import SimpleITK as sitk
    except ImportError:
        logger.error('requires "SimpleITK" to write NIFTI')
        raise

    assert uri.endswith(".nii") or uri.endswith(".nii.gz"), "not an NIFTI URI"

    data = sitk.GetImageFromArray(data)
    sitk.WriteImage(data, uri)
