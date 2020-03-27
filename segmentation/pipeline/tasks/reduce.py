"""
This module provides reduction functions for probabilities.
"""

import numpy as np
from prefect import task

__all__ = ["probabilty_to_label"]


@task
def probabilty_to_label(probability, axis=0, dtype=np.uint8):
    """
    Convert probability array to label array by their index. First dimension is un-classified.

    Args:
        prob (array-like): the probability map
        axis (int, optional): the axis to caclulate the index
    """
    labels = probability[1:, ...].argmax(axis=axis)
    return labels.astype(dtype) + 1
