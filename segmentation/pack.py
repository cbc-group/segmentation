"""
Pack raw data and labels into HDF5 format.
"""
import glob
import logging
import os

import h5py
import imageio
import numpy as np
import SimpleITK as sitk

import datetime

logger = logging.getLogger(__name__)


def load_file_by_ext(src_dir, ext: str, reader, desc="data", return_path=False):
    paths = glob.glob(os.path.join(src_dir, f"*.{ext}"))
    try:
        if len(paths) > 1:
            raise RuntimeError(f"too many {desc} sources")
        path = paths[0]
    except IndexError:
        raise RuntimeError(f"unable to locate {desc}")

    data = reader(path)
    if return_path:
        return path, data
    else:
        return data


def load_label(src_dir, ext="nii.gz"):
    def reader(path):
        data = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(data)

    path, data = load_file_by_ext(src_dir, ext, reader, desc="label", return_path=True)

    logger.info(f'loading label from "{os.path.basename(path)}"')
    return data


def load_raw_data(src_dir, ext="nrrd"):
    path, data = load_file_by_ext(
        src_dir, ext, imageio.imread, desc="raw data", return_path=True
    )

    logger.info(f'loading raw from "{os.path.basename(path)}"')
    return data


def main(src_dir, dst_dir=None, no_label=False, split_along="z"):
    raw = load_raw_data(src_dir)
    logger.info(f"dataset shape {raw.shape}")

    if not no_label:
        for ext in ("nii.gz", "nii"):
            try:
                label = load_label(src_dir, ext=ext)
                break
            except RuntimeError:
                pass
        else:
            raise RuntimeError("unable to locate label")
        assert raw.shape == label.shape, "raw data and label should have the same shape"

        if label.dtype != np.uint8:
            logger.warning("force label to cast as uint8")
            label = label.astype(np.uint8)

    if dst_dir is None:
        ts = datetime.datetime.now().timestamp()
        ts = int(ts)
        dst_dir = os.path.join(src_dir, f"packed_{ts}")
    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        pass
    logger.info(f'saving result to "{dst_dir}"')

    # DEBUG remove partial data
    raw, label = raw[:54, ...], label[:54, ...]

    # split along depth
    axis = "zyx".index(split_along)
    ds = raw.shape[axis] // 2

    logger.info(".. training set")
    path = os.path.join(dst_dir, "train.h5")
    with h5py.File(path, "w") as h:
        h["raw"] = raw[:ds, ...]
        h["label"] = label[:ds, ...]

    logger.info(".. validation set")
    path = os.path.join(dst_dir, "val.h5")
    with h5py.File(path, "w") as h:
        h["raw"] = raw[ds:, ...]
        h["label"] = label[ds:, ...]


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main(
        src_dir="U:/Andy/20200217_K8_nkcc2_568_10x15_z5um_1/glomerulus", split_along="y"
    )
