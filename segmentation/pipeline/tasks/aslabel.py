import glob
import logging
import os

import click
import coloredlogs
import h5py
import imageio
import numpy as np
import SimpleITK as sitk
from dask import delayed
from dask.distributed import Client, as_completed
from tqdm import tqdm

from ..steps import pack_arrays
from .utils import create_dir

__all__ = ["main"]

logger = logging.getLogger("segmentation.pipeline.tasks")


@delayed
def run(h5_path, dst_dir):
    print("load probability map")
    with h5py.File(h5_path, "r") as h:
        predictions = np.array(h["predictions"])

    labels = np.argmax(predictions[1:, ...], axis=0)
    labels = labels.astype(np.uint16) + 1  # match raw data data type

    labels = sitk.GetImageFromArray(labels)
    base_path = h5_path.rsplit("_", 1)
    tiff_path = base_path[0] + "_labels.tif"
    path = os.path.join(dst_dir, tiff_path)
    sitk.WriteImage(labels, path)

    return os.path.basename(tiff_path)


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
    dname = dname.rsplit('_', 1)[0]
    dname = f"{dname}_labels"
    dst_dir = os.path.join(os.path.dirname(src_dir), dname)
    create_dir(dst_dir)

    # write back
    write_back_tasks = []
    for src_path in enumerate(files):
        path = client.submit(run, src_path, dst_dir)

        fname = f"tile_{i:04d}.h5"
        dst_path = os.path.join(dst_dir, fname)
        future = pack_arrays(dst_path, data)

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
