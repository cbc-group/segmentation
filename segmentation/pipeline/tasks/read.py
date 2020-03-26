from dask import delayed
import numpy as np


@delayed
def read_h5(h5_path, internal_path="/"):
    import h5py

    with h5py.File(h5_path, "r") as h:
        return np.array(h[internal_path])
