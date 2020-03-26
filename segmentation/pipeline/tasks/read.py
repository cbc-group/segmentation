from dask import delayed
import dask.array as da
import numpy as np


@delayed
def read_h5(h5_path, internal_path="/"):
    import h5py

    with h5py.File(h5_path, "r") as h:
        data = np.array(h[internal_path])
    print(f"{h5_path}, {data.shape}")
    return da.from_array(data, chunks="auto")
