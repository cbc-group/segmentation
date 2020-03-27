import logging
from dask import delayed

__all__ = ["write_tiff", "write_nifti", "write_h5", "write_zarr"]

logger = logging.getLogger("segmentation.pipeline.tasks")


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

    print(f"write_back={uri}")
    imageio.volwrite(uri, data)

    return uri


def write_nifti(uri: str, data):
    import SimpleITK as sitk

    assert uri.endswith(".nii") or uri.endswith(".nii.gz"), "not an NIFTI URI"

    data = sitk.GetImageFromArray(data)
    sitk.WriteImage(data, uri)

    return uri


def write_zarr(uri, path, data):
    import dask.array as da

    da.to_zarr(data, uri, path)

    return uri


def write_h5(uri, path, data):
    import h5py
    import dask.array as da

    # NOTE
    # This failed to work, causing h5py cannot be pickled error.
    #   can't pickle local object when doing a to_hdf5 when using dask.distributed
    #   https://github.com/dask/distributed/issues/927
    da.to_hdf5(uri, path, data)

    # logger.info(f"uri={uri}, path={path}, data={data.shape},{data.dtype}")
    # with h5py.File(uri, "w") as h5:
    #    dst = h5.create_dataset(path, shape=data.shape, dtype=data.dtype)
    #    store(data, dst)

    return uri
