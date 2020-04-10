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
    ("C:/Users/Andy/Desktop/segmentation/workspace/ground_truth.nii.gz", (57, None)),
    ("D:/ground_truth.nii", (0, 57)),
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
sitk.WriteImage(label, "../workspace/ground_truth_merged.nii.gz")
sitk.WriteImage(label, "../workspace/ground_truth_merged.tif")
