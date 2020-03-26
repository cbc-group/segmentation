from dask import delayed
import dask.array as da


@delayed
def read_h5(h5_path, internal_path="/"):
    import h5py

    with h5py.File(h5_path, "r") as h:
        return da.from_array(h[internal_path])
