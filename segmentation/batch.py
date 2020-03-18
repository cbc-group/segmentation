"""
Process the entire dataset.
"""
import logging
import os

import click
import coloredlogs
import yaml

import torch
from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.predict import _get_predictor
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.model import get_model
from utoolbox.io.dataset import open_dataset
from typing import List
from dask.distributed import Client
import dask.bag as db

logger = logging.getLogger(__name__)


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


def run(config, tiles):
    """
    The worker function that process the tiles.
    """

    # load model
    model = load_model(config)

    # downsample tiles and write them to scratch
    for tid, tile in enumerate(tiles):
        pass

    # update config path

    for test_loader in get_test_loaders(config):
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")

        output_file = _get_output_file(test_loader.dataset)

        predictor = _get_predictor(model, test_loader, output_file, config)
        # run the model prediction on the entire dataset and save to the 'output_file' H5
        predictor.predict()


@click.command()
@click.argument("config_path")
@click.argument("src_dir")
def main(config_path, src_dir):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    # assume we have tunnel the scheduler to local
    client = Client("localhost:8786")
    print(client)

    # load config
    config = load_config(config_path)

    # load dataset
    src_ds = open_dataset(src_dir)
    desc = tuple(
        f"{k}={v}" for k, v in zip(("x", "y", "z"), reversed(src_ds.tile_shape))
    )
    logger.info(f"tiling dimension ({', '.join(desc)})")

    # retrieve tiles
    def retrieve(tile):
        data = src_ds[tile]

        sampler = (slice(None, None, 4),) * 2  # TODO fixed value to 4
        # TODO I know this is a 3D stack
        # normally, we don't sub-sample z
        sampler = (slice(None, None, None),) + sampler

        data = data[sampler]

        return data

    # generate tile index list (TODO deal with multi-color/view here)
    def groupby_tiles(inventory, index: List[str]):
        """
        Aggregation function that generates the proper internal list layout for all the tiles in their natural N-D layout.

        Args:
            inventory (pd.DataFrame): the listing inventory
            index (list of str): the column header
        """
        tiles = []
        for _, tile in inventory.groupby(index[0]):
            if len(index) > 1:
                # we are not at the fastest dimension yet, decrease 1 level
                tiles.extend(groupby_tiles(tile, index[1:]))
            else:
                # fastest dimension, call retrieval function
                tiles.append(retrieve(tile))
        return tiles

    index = ["tile_y", "tile_x"]
    if "tile_z" in src_ds.index.names:
        index = ["tile_z"] + index
    logger.info(f"a {len(index)}-D tiled dataset")

    raise RuntimeError("DEBUG")

    tiles = groupby_tiles(src_ds, index)
    logger.info(f"{len(tiles)} to process")
    tiles = db.from_delayed(tiles)


if __name__ == "__main__":
    main(
        "../configs/tubule/test_config_ce.yaml",
        "~/data/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1",
    )
