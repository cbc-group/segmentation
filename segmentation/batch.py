"""
Process the entire dataset.
"""
import logging
import os
from typing import List

import click
import coloredlogs
import imageio
from dask import delayed
from dask.distributed import Client, progress, as_completed
from utoolbox.io.dataset import open_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_dir(path):
    try:
        os.makedirs(path)
        print(f'"{path}" created')
    except FileExistsError:
        print(f'"{path}" exists')


def load_model(config):
    from pytorch3dunet.unet3d.model import get_model
    from pytorch3dunet.unet3d import utils

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
    import torch
    import yaml

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


def run(config_path, tiles):
    """
    The worker function that process the tiles.
    """
    # delayed load
    from pytorch3dunet.datasets.utils import get_test_loaders
    from pytorch3dunet.predict import _get_predictor

    # load config
    config = load_config(config_path)

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
    scheduler = "localhost:8786"
    logger.info(f'connecting to scheduler at "{scheduler}"')
    client = Client(scheduler, timeout=None)
    print(client)

    src_dir = os.path.abspath(src_dir)

    # load dataset
    src_ds = open_dataset(src_dir)
    desc = tuple(
        f"{k}={v}" for k, v in zip(("x", "y", "z"), reversed(src_ds.tile_shape))
    )
    logger.info(f"tiling dimension ({', '.join(desc)})")

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
                tiles.append(src_ds[tile])
        return tiles

    index = ["tile_y", "tile_x"]
    if "tile_z" in src_ds.index.names:
        index = ["tile_z"] + index
    logger.info(f"a {len(index)}-D tiled dataset")

    tiles = groupby_tiles(src_ds, index)
    logger.info(f"{len(tiles)} to process")

    # downsample
    tiles_bin4 = [tile[:, ::4, ::4] for tile in tiles]
    tiles_bin4 = [tiles_bin4[128]]  # DEBUG

    dname = os.path.basename(src_dir)
    dname = f"{dname}_bin4"
    dst_dir = os.path.join(os.path.dirname(src_dir), dname)
    create_dir(dst_dir)

    # write back
    write_back_tasks = []
    for i, tile in enumerate(tiles_bin4):
        fname = f"tile_{i:04d}.tif"
        path = os.path.join(dst_dir, fname)
        future = delayed(imageio.volwrite)(path, tile)
        write_back_tasks.append(future)
    futures = client.compute(write_back_tasks, scheduler="processes")

    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            print(future.result())
            pbar.update(1)

    # NOTE timeout has to have a number (in this case, default to 5min), otherwise,
    # no_default triggered causing TyperError
    logger.info("closing scheduler connection")
    client.close(timeout=300)


if __name__ == "__main__":
    main()
