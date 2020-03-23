import glob
import logging
import os

import click
import coloredlogs
import imageio
from dask import delayed
from dask.distributed import Client, as_completed
from tqdm import tqdm

from ..steps import pack_arrays
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

    files = glob.glob(os.path.join(src_dir, "*.tif"))
    logger.info(f"{len(files)} tile(s) to convert")

    dname = os.path.basename(src_dir)
    dname = f"{dname}_h5"
    dst_dir = os.path.join(os.path.dirname(src_dir), dname)
    create_dir(dst_dir)

    # write back
    write_back_tasks = []
    for i, path in enumerate(files):
        fname = f"tile_{i:04d}.h5"
        path = os.path.join(dst_dir, fname)
        data = delayed(imageio.volread)(path)
        future = pack_arrays(path, data)
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
