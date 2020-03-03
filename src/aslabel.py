"""
Convert probability maps into NIFTI label.
"""
import logging

import coloredlogs
import h5py
import numpy as np
import SimpleITK as sitk


logger = logging.getLogger(__name__)


coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

h5 = "C:/Users/Andy/Desktop/segmentation/data/train_predictions.h5"

logger.info("load probability maps")
with h5py.File(h5, "r") as h:
    predictions = np.array(h["predictions"])
logger.info(f"found {predictions.shape[0]} class(es)")

logger.info("apply argmax")
labels = np.argmax(predictions, axis=0)  # along the classes
labels = labels.astype(np.uint8)

logger.info("load labels")
labels = sitk.GetImageFromArray(labels)
sitk.WriteImage(labels, "predictions.nii.gz")
