from prefect import task
import numpy as np


@task
def read_h5(h5_path, internal_path="/"):
    import h5py

    with h5py.File(h5_path, "r") as h:
        return np.array(h[internal_path])
