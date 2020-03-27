import logging
from prefect import task

__all__ = ["write_tiff", "write_nifti", "write_h5"]

logger = logging.getLogger("segmentation.pipeline.tasks")


@task
def write_tiff(uri, data):
    """
    Write TIFF.

    Args:
        uri (str): target path
        data (array-like): the data
    
    Returns:
        (Future)
    """
    import imageio

    imageio.volwrite(uri, data)

    return uri


@task
def write_nifti(uri: str, data):
    import SimpleITK as sitk

    assert uri.endswith(".nii") or uri.endswith(".nii.gz"), "not an NIFTI URI"

    data = sitk.GetImageFromArray(data)
    sitk.WriteImage(data, uri)

    return uri


@task
def write_h5(uri, path, data):
    import dask.array as da

    da.to_hdf5(uri, path, data)

    return uri
