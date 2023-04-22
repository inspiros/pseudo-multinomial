import numpy as np

__all__ = ['consecutive_bincount']


def consecutive_bincount(x, minlength=None, average=True):
    """
    Count all consecutive occurrences.
    """
    x = x.astype(np.int64)
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
