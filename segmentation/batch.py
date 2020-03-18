"""
Process the entire dataset.
"""
import logging

import click
import coloredlogs
import yaml

import torch
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.model import get_model
from utoolbox.io.dataset import open_dataset

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


@click.command()
@click.argument("config_path")
@click.argument("src_dir")
def main(config_path, src_dir):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    # load config
    config = load_config(config_path)

    # load dataset
    src_ds = open_dataset(src_dir)
    desc = tuple(
        f"{k}={v}" for k, v in zip(("x", "y", "z"), reversed(src_ds.tile_shape))
    )
    logger.info(f"tiling dimension ({', '.join(desc)})")

    # load model
    model = load_model(config)

    

if __name__ == "__main__":
    main()
