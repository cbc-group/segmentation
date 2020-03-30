import glob
import logging
import os
import sys
from functools import partial
from itertools import repeat

from dask import delayed
import dask.bag as db
import dask.array as da
import h5py
import zarr
from dask.distributed import Client, get_client, progress, wait

from segmentation.pipeline.tasks import read_tiff, write_zarr, downsample_naive

logger = logging.getLogger(__name__)


def find_src_files(src_dir, file_ext):
    search_at = os.path.join(src_dir, f"*.{file_ext}")
    logger.info(f'search at "{search_at}"')

    tiff_paths = glob.glob(search_at)
    logger.info(f"found {len(tiff_paths)} files")

    return db.from_sequence(tiff_paths, partition_size=5)


def create_dst_dir(dst_dir):
    try:
        os.mkdir(dst_dir)
    except FileExistsError:
        logger.warning(f'"{dst_dir}" already exists')
    return dst_dir


def _build_path(dst_dir, src_path, file_ext):
    fname = os.path.basename(src_path)
    fname, _ = os.path.splitext(fname)
    fname = f"{fname}.{file_ext}"
    return os.path.join(dst_dir, fname)


def build_zarr_path(dst_dir, tiff_path):
    return _build_path(dst_dir, tiff_path, "zarr")


def build_h5_path(dst_dir, zarr_path):
    return _build_path(dst_dir, zarr_path, "h5")


def convert_hdf5(zarr_src, h5_dst):
    source = zarr.open(zarr_src, "r")

    with h5py.File(h5_dst, mode="w") as dest:
        zarr.copy_all(source, dest, log=sys.stdout, if_exists="replace")


def run(src_dir, dst_dir):
    client = get_client()

    # load data
    tiff_paths = find_src_files(src_dir, "tif")
    raw_data = tiff_paths.map(read_tiff)

    # create destination
    create_dst_dir(dst_dir)

    # # downsample
    # bin4_data = raw_data.map(partial(downsample_naive, ratio=(1, 4, 4)))
    # bin4_data = bin4_data.map(da.rechunk)
    # bin4_data = client.persist(bin4_data)
    #
    # logger.info("downsampling")
    # progress(bin4_data)

    logger.info("persist data on cluster")
    bin4_data = client.persist(raw_data, priority=-10)
    progress(bin4_data)

    # save intermediate result
    zarr_paths = tiff_paths.map(partial(build_zarr_path, dst_dir))
    name_data = db.zip(zarr_paths, bin4_data)
    futures = name_data.starmap(write_zarr, path="raw")

    logger.info("save as zarr")
    future = client.compute(futures, priority=10)
    progress(future)

    # convert to h5 for ingestion
    h5_paths = zarr_paths.map(partial(build_h5_path, dst_dir))
    src_dst = db.zip(zarr_paths, h5_paths)
    futures = src_dst.starmap(convert_hdf5)

    logger.info("convert zarr to h5")
    future = client.compute(futures, priority=20)
    progress(future)


def main():
    # client = Client("localhost:8786")
    client = Client(n_workers=4, threads_per_worker=4)

    root = "U:/Andy/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1_bin4"
    run(
        src_dir=os.path.join(root, "raw"), dst_dir=os.path.join(root, "bin4_h5"),
    )

    client.close()


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
