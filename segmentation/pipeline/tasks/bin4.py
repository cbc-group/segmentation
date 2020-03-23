import logging
import os
from typing import List

import click
import coloredlogs
from dask.distributed import Client, as_completed
from tqdm import tqdm
from utoolbox.io.dataset import open_dataset

from ..steps import downsample_naive, write_tiff
from .utils import create_dir

__all__ = ["main"]

logger = logging.getLogger("segmentation.pipeline.tasks")


@click.command()
@click.argument("src_dir", type=click.Path(file_okay=False, dir_okay=True, exists=True))
def main(src_dir):
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

    tiles = [tiles[0]]
    tiles_bin4 = [downsample_naive(tile, 4) for tile in tiles]

    dname = os.path.basename(src_dir)
    dname = f"{dname}_bin4"
    dst_dir = os.path.join(os.path.dirname(src_dir), dname)
    create_dir(dst_dir)

    # write back
    write_back_tasks = []
    for i, tile in enumerate(tiles_bin4):
        fname = f"tile_{i:04d}.tif"
        path = os.path.join(dst_dir, fname)
        future = write_tiff(path, tile)
        write_back_tasks.append(future)

    # submit task
    futures = client.compute(write_back_tasks, scheduler="processes")
    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            pbar.update(1)

    logger.info("closing scheduler connection")
    client.close()


if __name__ == "__main__":
    main()
