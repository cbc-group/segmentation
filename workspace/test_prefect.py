import glob
import logging
import os

from dask.distributed import Client
from prefect import Flow, Parameter, task, unmapped
from prefect.engine.executors import DaskExecutor
from prefect.tasks.core.constants import Constant

from segmentation.pipeline.tasks import read_tiff, write_h5

logger = logging.getLogger("segmentation.workspace")


@task
def find_src_files(src_dir):
    tiff_paths = glob.glob(os.path.join(src_dir, "*.tif"))
    logger.info(f"found {len(tiff_paths)} files")
    return tiff_paths


@task
def build_h5_path(tiff_path):
    src_dir, fname = os.path.split(tiff_path)
    print(src_dir)
    src_dir = f"{src_dir}_h5"
    fname, _ = os.path.splitext(fname)
    fname = f"{fname}.h5"
    return os.path.join(src_dir, fname)


def main():
    with Flow("inference") as flow:
        src_dir = Parameter("src_dir")
        tiff_paths = find_src_files(src_dir)

        logger.info("loading raw data")
        raw_data = read_tiff.map(tiff_paths)

        logger.info("dumping to hdf5")
        h5_paths, path = build_h5_path.map(tiff_paths), Constant("raw")
        write_h5.map(h5_paths, unmapped(path), raw_data)

        logger.info("inference")

    client = Client("localhost:8786")
    executor = DaskExecutor(address=client.scheduler.address)

    flow.run(
        src_dir="/home/ytliu/data/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1_bin4",
        executor=executor,
    )


if __name__ == "__main__":
    main()
