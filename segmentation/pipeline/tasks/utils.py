import os
import logging

__all__ = ["create_dir"]

logger = logging.getLogger("segmentation.pipeline.tasks")


def create_dir(path):
    try:
        os.makedirs(path)
        logger.info(f'"{path}" created')
    except FileExistsError:
        logger.warning(f'"{path}" exists')
