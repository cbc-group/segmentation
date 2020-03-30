from typing import Union, Tuple
from functools import lru_cache

__all__ = ["downsample_naive"]


@lru_cache(maxsize=None)
def _create_sampler(ratio, ndim):
    assert len(ratio) == ndim, "ratio elements should match the dimensions"
    return tuple(slice(None, None, r) for r in ratio)


def downsample_naive(data, ratio: Union[int, Tuple[int]]):
    """
    Using element-skip to downsample the data.

    Args:
        data (array-like): the data
        ratio (int): downsampling ratio

    Returns:
        (array-like): downsampled data
    """
    # inflate to n-dim tuple first
    if not isinstance(ratio, tuple):
        ratio = (ratio,) * data.ndim

    # test data type
    assert ratio[0] >= 1, "downsampling ratio should be >= 1"
    assert isinstance(ratio[0], int), "non-integer step is invalid"

    samplers = _create_sampler(ratio, data.ndim)

    return data[samplers]
