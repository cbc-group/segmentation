import glob
import logging
import os
from typing import List

import dask.array as da
import h5py
import imageio
import numpy as np
import prefect
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
    logger.info(f"found {len(tiff_paths)} files")

    return tiff_paths


def read_h5(h5_path):
    with h5py.File(h5_path, mode="r") as h:
        return np.array(h["predictions"])


@task
def preload_array_info(paths: List[str]):
    assert len(paths) > 0, "no reference file exist"

    with h5py.File(paths[0], mode="r") as h:
        # NOTE `read_h5` is not used here, since we don't need to load the array to
        # find out shape, dtype info
        data = h["predictions"]
        shape, dtype = data.shape, data.dtype
    prefect.context.logger.info(f"preload array {shape}, {dtype}")

    return shape, dtype


@task
def read_prob_map(h5_path, array_info):
    shape, dtype = array_info

    with h5py.File(h5_path, mode="r") as h:
        return da.from_array(h["predictions"])


def create_dir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        logger.warning(f'"{path}" already exists')
    return path


@task
def classify(data, dtype=np.uint16):
    label = data[1:, ...].argmax(axis=0)
    return label.astype(dtype) + 1


@task
def build_path(dst_dir, src_path, file_ext: str):
    fname = os.path.basename(src_path)
    fname, _ = os.path.splitext(fname)
    fname = f"{fname}.{file_ext}"
    return os.path.join(dst_dir, fname)


@task
def write_tiff(uri, data):
    with imageio.get_writer(uri) as writer:
        writer.set_meta_data({"compress": 6})
        data = data.compute()
        for im in data:
            writer.append_data(im)

    return uri


def run(src_dir, dst_dir, debug=False):
    src_dir = Parameter("src_dir", src_dir)

    # create destination
    create_dir(dst_dir)
    dst_dir = Parameter("dst_dir", dst_dir)

    with Flow("classify_pipeline") as flow:
        # load data
        h5_paths = find_src_files(src_dir, "h5")
        info = preload_array_info(h5_paths)
        prob_map = read_prob_map.map(h5_paths, unmapped(info))

        # classify
        label = classify.map(prob_map)

        # save
        tiff_paths = build_path.map(unmapped(dst_dir), h5_paths, unmapped("tif"))
        write_tiff.map(tiff_paths, label)

    if debug:
        flow.visualize()
    else:
        client = get_client()
        executor = DaskExecutor(address=client.scheduler.address)

        flow.run(executor=executor)


def main():
    # client = Client(n_workers=4, threads_per_worker=4)
    # root = "U:/Andy/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1_bin4"

    client = Client(scheduler_file="/home/ytliu/scheduler.json")
    root = "/home/ytliu/data/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1_process"
    run(
        src_dir=os.path.join(root, "prob_map"), dst_dir=os.path.join(root, "label"),
    )

    client.close()


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
