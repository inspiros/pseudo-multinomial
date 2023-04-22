# distutils: language = c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional

import cython
cimport numpy as cnp
import numpy as np
from cpython cimport array
from libc cimport math

from .fptr cimport DoubleSeriesFPtr, PyDoubleSeriesFPtr
from ..utils.random_utils cimport mt19937, seed_mt19937, get_10_rand_bits

__all__ = [
    'shanks',
]

cdef double MACHINE_EPS = <double> np.finfo(np.float64).eps

cdef inline unsigned int min(unsigned int a, unsigned int b) nogil:
    return a if a < b else b

cdef inline unsigned int max(unsigned int a, unsigned int b) nogil:
    return a if a > b else b

# --------------------------------
# Shanks transformation
# --------------------------------
# noinspection DuplicatedCode
cdef cnp.ndarray[cnp.float64_t, ndim=2] wynn_eps_kernel(
        DoubleSeriesFPtr f,
        long start = 0,
        double start_val = 0,
        unsigned int max_r = 0,
        double atol = 1e-14,
        unsigned int max_iter = 200,
        unsigned int extend_step = 50,
        int zero_div = 0):
    extend_step = max(extend_step, 1)
    cdef:
        unsigned int max_ncols = (max_r + 1) * 2 + 1 if max_r > 0 else 0
        unsigned int init_ncols = max_iter if max_iter > 0 else extend_step
        array.array double_array_template = array.array('d', [])
        array.array int_array_template = array.array('I', [])
        cnp.ndarray[cnp.float64_t, ndim=2] rows = np.empty((2, init_ncols), dtype=np.float64)
        cnp.ndarray[cnp.float64_t, ndim=2] eps = np.empty((3, init_ncols), dtype=np.float64)

    rows[0, 0] = start_val + f.eval(start)
    cdef double term = f.eval(start + 1)
    cdef double prev_term = term
    rows[1, 0] = rows[0, 0] + term
    eps[1, 0] = term
    eps[0, 0] = rows[1, 0]
    eps[2, 0] = 1
    if math.fabs(term) <= atol:
        return eps[:, :1]

    eps[:, 1:] = math.NAN
    rows[0, 0] = rows[1, 0]
    rows[0, 1] = 1 / term
    rows[:, 2:] = math.NAN

    cdef:
        unsigned int i, j, r, ncols, ncols_used
        double denom
        bint force_return = False
    i = 2
    ncols = i + 1
    ncols_used = ncols
    while True:
        if i >= max_iter > 0:
            break
        term = f.eval(start + i)
        rows[1, 0] = rows[0, 0] + term

        eps[1, 0] = term
        eps[0, 0] = rows[1, 0]
        eps[2, 0] = i
        if not math.isfinite(term) or math.fabs(term) <= atol:
            break
        elif term / prev_term >= 1:
            if math.copysign(1, term) * math.copysign(1, prev_term) > 0:
                eps[0, 0] = math.copysign(math.INFINITY, eps[0, 0])
            else:
                eps[0, 0] = math.NAN
            break
        rows[1, 1] = 1 / term  # first dummy column

        j = 2
        while j < ncols:
            denom = rows[1, j - 1] - rows[0, j - 1]
            if denom == 0:
                if zero_div < 2:
                    rows[1, j:ncols] = math.NAN
                    ncols = j
                    if zero_div == 0:  # return
                        force_return = True
                    break
                else:  # random
                    denom = (<double> get_10_rand_bits() + 1) * MACHINE_EPS
            rows[1, j] = rows[0, j - 2] + 1 / denom
            if not j & 1:
                r = j // 2
                eps[1, r] = rows[1, j] - eps[0, r]
                eps[0, r] = rows[1, j]
                eps[2, r] = i
            j += 1
        # advance
        rows[0, :ncols] = rows[1, :ncols]
        i += 1
        ncols = min(ncols + 1, max_ncols) if max_ncols > 0 else ncols + 1
        ncols_used = max(ncols, ncols_used)
        if force_return:
            break
        if max_iter == 0 and ncols_used > rows.shape[1]:  # extend
            rows = np.pad(rows, ((0, 0), (0, extend_step)),
                          constant_values=math.NAN)
            eps = np.pad(eps, ((0, 0), (0, extend_step)),
                         constant_values=math.NAN)
    return eps[:, :ncols_used // 2]

# noinspection DuplicatedCode
@cython.binding(True)
def shanks(f: Callable[[int], float],
           start: int = 0,
           start_val: float = 0,
           max_r: Optional[int] = None,
           atol: float = 1e-14,
           max_iter: Optional[int] = 200,
           extend_step: int = 50,
           zero_div: str = 'ignore',
           return_table: bool = False):
    """Use Shanks transformation to find sum of series.

    Args:
        f: 
        start: 
        start_val: 
        max_r: 
        atol: 
        max_iter: 
        zero_div: 
        extend_step: 
        return_table: 

    Returns:
    """
    # check args
    if max_r is None or max_r < 0:
        max_r = 0
    if atol < 0:
        raise ValueError('atol must be non-negative. '
                         f'Got {atol}.')
    if max_iter is None:
        max_iter = 0
    if zero_div not in ['return', 'ignore', 'random']:
        raise ValueError('Supported zero_div strategies are '
                         '\'return\', \'ignore\', and \'random\'. '
                         f'Got {zero_div}.')
    zero_div_map = {
        'return': 0,
        'ignore': 1,
        'random': 2,
    }

    f_wrapper = PyDoubleSeriesFPtr.from_f(f)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] table = wynn_eps_kernel(
        f_wrapper, start, start_val, max_r, atol, max_iter, zero_div_map[zero_div], extend_step)

    if return_table:
        return table
    eps, diffs, steps = table
    if not math.isfinite(eps[0]):  # diverged
        return eps[0]
    elif diffs[0] == 0:  # converged
        return eps[0]
    elif np.all(steps[-1] >= steps[:-1]):  # highest order shanks
        return eps[-1]
    # otherwise, return the column with the least change
    diffs = np.nan_to_num(diffs, nan=np.inf)
    diffs[0] = np.inf
    return eps[np.argmin(np.abs(diffs))]

@cython.binding(True)
cpdef cnp.ndarray[cnp.float64_t, ndim=2] wynn_eps(sn: double[:],
                                                  r: int = None,
                                                  randomized: bint = False):
    """Perform Wynn Epsilon Convergence Algorithm"""
    if r is None:
        r = (sn.shape[0] - 1) // 2
    else:
        r = min(r, (sn.shape[0] - 1) // 2)
    cdef long n = 2 * r + 1

    cdef:
        double[:, :] e = np.empty(shape=(sn.shape[0], n))
        long i, j
        double denom

    with nogil:
        e[:, 0] = sn[:]
        e[:, 1:] = math.NAN
        for i in range(1, sn.shape[0]):  # i = n
            denom = e[i, 0] - e[i - 1, 0]
            if denom == 0:
                break
            e[i, 1] = 1 / denom  # first dummy column

            for j in range(2, min(i + 2, n)):  # j = r + 1
                denom = e[i, j - 1] - e[i - 1, j - 1]
                if denom == 0:
                    if randomized:
                        denom = (<double> get_10_rand_bits() + 1) * MACHINE_EPS
                    else:
                        break
                if math.isnan(denom):
                    break
                e[i, j] = e[i - 1, j - 2] + 1 / denom
    return np.asarray(e[:, ::2])
