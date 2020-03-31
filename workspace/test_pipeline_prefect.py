import glob
import logging
import os
import sys

import dask.array as da
import h5py
import imageio
import prefect
import zarr
from dask import delayed
from dask.distributed import Client, get_client
from prefect import Flow, Parameter, task, unmapped
from prefect.engine.executors import DaskExecutor

logger = logging.getLogger(__name__)


@task
def find_src_files(src_dir, file_ext: str = "*"):
    search_at = os.path.join(src_dir, f"*.{file_ext}")
    logger.info(f'search at "{search_at}"')

    tiff_paths = glob.glob(search_at)
    tiff_paths = tiff_paths[:5]  # DEBUG
    logger.info(f"found {len(tiff_paths)} files")

    return tiff_paths


@task
def read_tiff(tiff_path):
    try:
        shape, dtype = prefect.context["shape"], prefect.context["dtype"]
    except KeyError:
        logger = prefect.context.logger
        logger.info("preloading shape and dtype")

        data = imageio.volread(tiff_path)
        shape, dtype = data.shape, data.dtype
        prefect.context["shape"], prefect.context["dtype"] = shape, dtype

    data = delayed(imageio.volread)(tiff_path)
    data = da.from_delayed(
        data, shape=shape, dtype=dtype, name=os.path.basename(tiff_path)
    )
    return data


def create_dir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        logger.warning(f'"{path}" already exists')
    return path


@task
def build_path(dst_dir, src_path, file_ext: str):
    fname = os.path.basename(src_path)
    fname, _ = os.path.splitext(fname)
    fname = f"{fname}.{file_ext}"
    return os.path.join(dst_dir, fname)


@task
def write_zarr(uri, data, internal_path="/"):
    data = data.rechunk("auto")
    da.to_zarr(data, uri, component=internal_path, overwrite=True)
    return uri


@task
def zarr_to_h5(zarr_path, h5_path):
    source = zarr.open(zarr_path, "r")

    with h5py.File(h5_path, mode="w") as dest:
        zarr.copy_all(source, dest, log=sys.stdout, if_exists="replace")


def run(src_dir, dst_dir, debug=True):
    src_dir = Parameter("src_dir", src_dir)

    # create destination
    create_dir(dst_dir)
    dst_dir = Parameter("dst_dir", dst_dir)

    with Flow("test_pipeline") as flow:
        # load data
        tiff_paths = find_src_files(src_dir, "tif")
        raw_data = read_tiff.map(tiff_paths)

        # save as zarr for faster access
        zarr_paths = build_path.map(unmapped(dst_dir), tiff_paths, unmapped("zarr"))
        zarr_paths = write_zarr.map(zarr_paths, raw_data, unmapped("raw"))

        # convert
        h5_paths = build_path.map(unmapped(dst_dir), zarr_paths, unmapped("h5"))
        zarr_to_h5.map(zarr_paths, h5_paths)

    if debug:
        flow.visualize()
    else:
        client = get_client()
        executor = DaskExecutor(address=client.scheduler.address)

        flow.run(executor=executor)


def main():
    client = Client("localhost:8786")
    # client = Client(n_workers=4, threads_per_worker=4)

    # root = "U:/Andy/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1_bin4"
    root = "/home/ytliu/data/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1_process"
    run(
        src_dir=os.path.join(root, "raw"), dst_dir=os.path.join(root, "h5"),
    )

    client.close()


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
