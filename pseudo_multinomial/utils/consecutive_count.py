from typing import Optional

import numpy as np

from .._types import VectorLike

__all__ = ['consecutive_bincount']


def consecutive_bincount(x: VectorLike, minlength: Optional[int] = None, average: bool = True):
    r"""
    Count all consecutive occurrences in a 1D array.
    """
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 1:
        raise ValueError(f'x must be a 1D array. Got x.shape={x.shape}.')

    change_inds = np.where(np.pad(x[:-1] != x[1:], (1, 1), constant_values=1))[0]
    vals = x[change_inds[:-1]]
    counts = np.diff(change_inds)
    unique_vals = np.unique(vals)

    if minlength:
        n = max(minlength, unique_vals.max() + 1)
    else:
        n = unique_vals.max() + 1
    res = np.zeros(n, dtype=np.float64)
    c = np.zeros_like(res)
    for i in range(counts.shape[0]):
        v = vals[i]
        res[v] += counts[i]
        c[v] += 1
    if average:
        res /= c
    return res
