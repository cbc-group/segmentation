import glob
import logging
import os

import click
import coloredlogs
import numpy as np
from dask.distributed import Client, as_completed
from tqdm import tqdm

from ..tasks import as_label, read_h5, write_tiff
from .utils import create_dir

__all__ = ["main"]

logger = logging.getLogger("segmentation.pipeline.flows")


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

    files = glob.glob(os.path.join(src_dir, "*_predictions.h5"))
    logger.info(f"{len(files)} tile(s) to convert")

    dname = os.path.basename(src_dir)
    dname = dname.rsplit("_", 1)[0]
    dname = f"{dname}_labels"
    dst_dir = os.path.join(os.path.dirname(src_dir), dname)
    create_dir(dst_dir)

    futures = []
    for f in files:
        probabilities = read_h5(f, "predictions")

        label = as_label(probabilities, dtype=np.uint16)

        fname = os.path.basename(f)
        fname, _ = os.path.splitext(fname)
        fname = f"{fname}.tif"
        tiff_path = os.path.join(dst_dir, fname)
        future = write_tiff(tiff_path, label)

        futures.append(future)

    futures = client.submit(futures)
    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                logger.exception(error)
            finally:
                pbar.update(1)
                del future

    logger.info("closing scheduler connection")
    client.close()


if __name__ == "__main__":
    main()
