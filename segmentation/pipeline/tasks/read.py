import logging

import dask.array as da
from dask.distributed import get_client
from prefect import task

__all__ = ["read_h5", "read_nifti"]

logger = logging.getLogger("segmentation.pipeline.tasks")


@task
def read_h5(uri, path="/"):
    import h5py

    client = get_client()
    with h5py.File(uri, "r") as h:
        data = da.from_array(h[path], chunks="auto")
        data = client.persist(data)

    return data


@task
def read_nifti(uri):
    import SimpleITK as sitk

    data = sitk.ReadImage(uri)
    data = sitk.GetArrayFromImage(data)

    client = get_client()
    data = da.from_array(data, chunks="auto")
    data = client.persist(data)

    return data


@task
def read_tiff(uri):
    import imageio

    data = imageio.volread(uri)

    client = get_client()
    data = da.from_array(data, chunks="auto")
    data = client.persist(data)

    return data
