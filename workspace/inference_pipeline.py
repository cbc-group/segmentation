import glob
import logging
import os
import sys
from itertools import chain
from typing import List

import dask.array as da
import h5py
import imageio
import prefect
import yaml
import zarr
from dask import delayed
from dask.distributed import Client, get_client
from prefect import Flow, Parameter, task, unmapped
from prefect.engine.executors import DaskExecutor

import torch
from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.predict import _get_predictor
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.model import get_model

logger = logging.getLogger(__name__)


@task
def find_src_files(src_dir, file_ext: str = "*"):
    search_at = os.path.join(src_dir, f"*.{file_ext}")
    logger.info(f'search at "{search_at}"')

    tiff_paths = glob.glob(search_at)
    logger.info(f"found {len(tiff_paths)} files")

    return tiff_paths


@task
def preload_array_info(paths: List[str]):
    assert len(paths) > 0, "no reference file exist"

    data = imageio.volread(paths[0])
    shape, dtype = data.shape, data.dtype
    prefect.context.logger.info(f"preload array {shape}, {dtype}")

    prefect.context["shape"], prefect.context["dtype"] = shape, dtype
    return shape, dtype


@task
def partition_path_list(paths: List[str], partition_size: int):
    n_partition = len(paths) // partition_size
    prefect.context.logger.info(f"partition into {n_partition} blocks")

    parted = []
    for i in range(0, n_partition):
        parted.append(paths[i::n_partition])
    return parted


def create_dir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        logger.warning(f'"{path}" already exists')
    return path


def load_model(config):
    # create the model
    model = get_model(config)

    # load model state
    model_path = config["model_path"]
    logger.info(f"Loading model from {model_path}...")
    utils.load_checkpoint(model_path, model)

    device = config["device"]
    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    return model


def load_config(path):
    config = yaml.safe_load(open(path, "r"))

    # get device to train on
    device_str = config.get("device", None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warn("CUDA not available, using CPU")
            device_str = "cpu"
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config["device"] = device
    return config


@task
def infer(h5_paths: List[str], config_path: str, dst_dir):
    # load model
    config = load_config(config_path)
    model = load_model(config)

    # update file path
    config["loaders"]["test"]["file_paths"] = h5_paths

    prob_h5_paths = []
    for test_loader in get_test_loaders(config):
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")

        fname = os.path.basename(test_loader.dataset.file_path)
        output_file = os.path.join(dst_dir, fname)

        predictor = _get_predictor(model, test_loader, output_file, config)
        predictor.predict()

        prob_h5_paths.append(output_file)
    return prob_h5_paths


@task
def combine_path_list(parted: List[List[str]]):
    return list(chain.from_iterable(parted))


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


def run(src_dir, dst_dir, config_path: str, debug=False):
    src_dir = Parameter("src_dir", src_dir)

    # create destination
    create_dir(dst_dir)
    dst_dir = Parameter("dst_dir", dst_dir)

    # number of workers
    config_path = Parameter("config_path", config_path)

    with Flow("inference_pipeline") as flow:
        # list tiles
        tiff_paths = find_src_files(src_dir, "h5")
        parted_tiff_paths = partition_path_list(tiff_paths, 5)

        prob_paths = infer.map(
            parted_tiff_paths, unmapped(config_path), unmapped(dst_dir)
        )
        prob_paths = combine_path_list(prob_paths)

    if debug:
        flow.visualize(filename="flow_debug")
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
        src_dir=os.path.join(root, "h5"),
        dst_dir=os.path.join(root, "prob_map"),
        config_path="/home/ytliu/segmentation/configs/tubule/test_config_ce.yaml",
    )

    client.close()


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
