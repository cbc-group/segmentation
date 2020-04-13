"""
Merge parts of label files with each other.
"""
import logging
import os

import coloredlogs
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

# NOTE range is [start, end)
files = [
    ("U:/Andy/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1/nucleus/ground_truth.nii.gz", (81, None)),
    ("U:/Andy/20191210_ExM_kidney_10XolympusNA06_zp3_10x14_kb_R_Nkcc2_488_slice_8_1/nucleus/ground_truth/ground_truth_20200410.nii.gz", (0, 81)),
]

array = None
for path, (start, end) in files:
    data = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(data)

    sampler = slice(start, end)

    try:
        if array.shape != data.shape:
            raise RuntimeError(f'"{os.path.basename(path)}" has different shape')
    except AttributeError:
        array = np.zeros_like(data)

    array[sampler, ...] = data[sampler, ...]

logger.info("write merged label")
label = sitk.GetImageFromArray(array)
sitk.WriteImage(label, "ground_truth_merged.nii.gz")
sitk.WriteImage(label, "ground_truth_merged.tif")
