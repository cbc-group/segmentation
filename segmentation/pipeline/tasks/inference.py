"""
Process the entire dataset.
"""
import glob
import logging
import os
from typing import List

import click
import coloredlogs
import imageio
import yaml
from dask import delayed
from dask.distributed import Client, as_completed
from tqdm import tqdm
from utoolbox.io.dataset import open_dataset

import torch
from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.predict import _get_predictor
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.model import get_model
import dask.bag as db

from .utils import create_dir

__all__ = ["main"]

logger = logging.getLogger("segmentation.pipeline.tasks")


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


def _get_output_file(dataset, suffix="_predictions"):
    return f"{os.path.splitext(dataset.file_path)[0]}{suffix}.h5"


def run(paths, config_path):
    """
    The worker function that process the tiles.

    Args:
        paths (Iterator)
        config_path (str) 
    """
    # load config
    config = load_config(config_path)

    # load model
    model = load_model(config)

    # update config path
    config["loaders"]["test"]["file_paths"] = paths

    for test_loader in get_test_loaders(config):
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")

        output_file = _get_output_file(test_loader.dataset)

        predictor = _get_predictor(model, test_loader, output_file, config)
        # run the model prediction on the entire dataset and save to the 'output_file' H5
        predictor.predict()

        yield output_file


@click.command()
@click.argument("config_path")
@click.argument("src_dir")
def main(config_path, src_dir):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    # assume we have tunnel the scheduler to local
    scheduler = "localhost:8786"
    logger.info(f'connecting to scheduler at "{scheduler}"')
    client = Client(scheduler, timeout="300s")  # 5 min
    print(client)

    src_dir = os.path.abspath(src_dir)

    files = glob.glob(os.path.join(src_dir, "*.h5"))
    logger.info(f"{len(files)} tile(s) to convert")

    files = db.from_sequence(files, npartitions=5)  # number of gpus
    output_paths = files.map_partitions(run, config_path=config_path)
    futures = client.compute(output_paths, scheduler="processes")

    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            pbar.update(1)

    logger.info("closing scheduler connection")
    client.close()


if __name__ == "__main__":
    main()
