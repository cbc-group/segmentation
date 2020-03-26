import glob
import logging
import os

import click
import coloredlogs
import numpy as np
from dask.distributed import Client
from prefect import Flow, Parameter, task
from prefect.engine.executors import DaskExecutor
from tqdm import tqdm

from ..tasks import as_label, read_h5, write_tiff
from .utils import create_dir

__all__ = ["main"]

logger = logging.getLogger("segmentation.pipeline.flows")


@task
def get_predictions(src_dir):
    files = glob.glob(os.path.join(src_dir, "*_predictions.h5"))
    logger.info(f"{len(files)} tile(s) to convert")
    return files


@task
def build_dst_path(h5_path, dst_dir):
    fname = os.path.basename(h5_path)
    fname, _ = os.path.splitext(fname)
    fname = f"{fname}.tif"
    return os.path.join(dst_dir, fname)


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

    with Flow("as_label") as flow:
        h5_path = Parameter("h5_path")
        dst_dir = Parameter("dst_dir")

        probabilities = read_h5(h5_path, "predictions")

        label = as_label(probabilities, dtype=np.uint16)

        tiff_path = build_dst_path(h5_path, dst_dir)
        write_tiff(tiff_path, label)

    executor = DaskExecutor(address=client.scheduler.address)

    with tqdm(total=len(files)) as pbar:
        for f in files:
            pbar.set_description(os.path.basename(f))
            flow.run(parameters={"h5_path": f, 'dst_dir': dst_dir}, executor=executor)
            pbar.update(1)

    logger.info("closing scheduler connection")
    client.close()


if __name__ == "__main__":
    main()
