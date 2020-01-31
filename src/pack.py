"""
Pack raw data and labels into HDF5 format.
"""
import logging
import os

import coloredlogs
import h5py
import imageio
import numpy as np
import SimpleITK as sitk


logger = logging.getLogger(__name__)


logging.getLogger("tifffile").setLevel(logging.ERROR)
coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

src_dir = (
    "U:/Andy/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1"
)

logger.info("load raw")
raw = os.path.join(src_dir, "3-Pos_000_008.nrrd")
raw = imageio.imread(raw)

logger.info("load label")
label = os.path.join(src_dir, "kidney_segmentation.nii.gz")
label = sitk.ReadImage(label)
label = sitk.GetArrayFromImage(label)

assert raw.shape == label.shape
logger.info(f"dataset shape {raw.shape}")

logger.info("writing")
h5 = os.path.join(src_dir, "dataset.h5")
with h5py.File(h5, "w") as h:
    h["raw"] = raw
    h["label"] = label.astype(np.uint8)
