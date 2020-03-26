"""
This module provides reduction functions for probabilities.
"""

from prefect import task
import numpy as np


@task
def as_label(probability, axis=0, dtype=np.uint8):
    """
    Convert probability array to label array by their index. First dimension is un-classified.

    Args:
        prob (array-like): the probability map
        axis (int, optional): the axis to caclulate the index
    """
    labels = probability[1:, ...].argmax(axis=axis)
    labels = labels.astype(dtype) + 1
    return labels.compute()
