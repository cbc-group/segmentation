import glob
import logging
import os
from functools import partial
from itertools import repeat

import dask.bag as db
from dask import delayed
from dask.distributed import Client, progress, get_client
import zarr
from segmentation.pipeline.tasks import read_tiff, write_tiff, write_zarr

logger = logging.getLogger(__name__)


def find_src_files(src_dir):
    search_at = os.path.join(src_dir, "*.tif")
    logger.info(f'search at "{search_at}"')

    tiff_paths = glob.glob(search_at)
    tiff_paths = tiff_paths[:5]  # DEBUG
    logger.info(f"found {len(tiff_paths)} files")

    return db.from_sequence(tiff_paths)


def create_h5_dst_dir(dst_dir):
    try:
        os.mkdir(dst_dir)
    except FileExistsError:
        logger.warning(f'"{dst_dir}" already exists')
    return dst_dir


def build_h5_path(dst_dir, tiff_path):
    fname = os.path.basename(tiff_path)
    fname, _ = os.path.splitext(fname)
    fname = f"{fname}.zarr"
    return os.path.join(dst_dir, fname)


def convert_hdf5(dst_dir):
    search_at = os.path.join(dst_dir, "*.zarr")
    zarr_paths = glob.glob(search_at)
    for zarr_path in zarr_paths:
        source = zarr.open(zarr_path, "r")
        fname, _ = os.path.splitext(zarr_path)
        fname = f"{fname}.h5"
        print(os.path.basename(fname))
        zarr.copy(source["raw"], fname["raw"])


def run(src_dir, dst_dir):
    client = get_client()

    # load data
    tiff_paths = find_src_files(src_dir)
    raw_data = tiff_paths.map(read_tiff)

    print(raw_data)

    # create destination
    create_h5_dst_dir(dst_dir)

    # re-save
    h5_paths = tiff_paths.map(partial(build_h5_path, dst_dir))
    name_data = db.from_sequence(zip(h5_paths, repeat("raw"), raw_data))
    tasks = name_data.starmap(write_zarr)

    progress(client.compute(tasks))

    logger.info("convert from zarr to hdf5")
    convert_hdf5(dst_dir)


def main():
    # client = Client("localhost:8786")
    client = Client(n_workers=2, threads_per_worker=2)

    run(
        src_dir="u:/andy/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1_bin4/raw",
        dst_dir="u:/andy/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1_bin4/h5",
    )

    client.close()


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
