# distutils: language = c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

import numpy as np
from libc cimport math

from .series.extrapolation import shanks
from .series.fptr cimport DoubleSeriesFPtr
from .utils._warnings import warn_value
from .typing import Callable, VectorLike

__all__ = [
    'Chain',
    'FiniteChain',
    'InfiniteChain',
    'AbsorbingChain',
    'ForwardingChain',
    'ListChain',
    'LinearChain',
    'GeometricChain',
    'HarmonicChain',
    'TanhChain',
    'SigmoidChain',
    'ATanChain',
    'LambdaChain',
]

################################################################################
# Base classes
################################################################################
# noinspection DuplicatedCode
cdef class Chain:
    """Base Sequential Markov chain
    """
    def __init__(self, initial_state: int = 1):
        if initial_state < 1:
            raise ValueError(f'initial_state must be positive. Got {initial_state}.')
        self._initial_state = initial_state
        self._state = self._initial_state

    cdef inline void reset(self) nogil:
        self._state = self._initial_state

    cdef unsigned long state(self) nogil:
        return self._state

    cdef void set_state(self, unsigned long k) nogil:
        self._state = k

    cdef inline unsigned long next_state(self, double p) nogil:
        if not self._state:
            return self._state
        if p > self.exit_probability_(self._state):
            self._state += 1
        else:  # exit
            self._state = 0
        return self._state

    cdef double exit_probability_(self, unsigned long k) nogil:
        cdef double p
        with gil:
            p = self.exit_probability(k)
        return p

    cpdef double exit_probability(self, unsigned long k):
        """Return exit probability at state ``k``."""
        raise NotImplementedError

    cdef inline double linger_probability_(self, unsigned long k) nogil:
        return 1 - self.exit_probability_(k)

    cpdef double expectation(self):
        raise NotImplementedError

    cpdef double n_states(self):
        raise NotImplementedError

    cpdef bint is_finite(self):
        return math.isfinite(self.n_states())

    cpdef np.ndarray[np.float64_t, ndim=2] transition_matrix(self, unsigned long n=0):
        if self.is_finite():
            n = <unsigned long> (self.n_states() if not n else math.fmin(self.n_states(), n))
        elif n == 0:
            raise ValueError('Complete transition matrix is not available '
                             'for infinite chains.')
        cdef np.ndarray[np.float64_t, ndim=2] P = np.zeros((n + 1, n + 1), dtype=np.float64)
        cdef unsigned long j
        P[0, 0] = 1  # absorbing exit state
        for j in range(1, n + 1):
            P[j, 0] = self.exit_probability_(j + self._initial_state - 1)
        for j in range(1, n):
            P[j, j + 1] = 1 - P[j, 0]
        return P

    def __len__(self):
        return int(self.n_states())

    def __repr__(self):
        return f'{self.__class__.__name__}()'

# noinspection DuplicatedCode
cdef class _ExitProbabilityCumProd(DoubleSeriesFPtr):
    """
    Utility class for computing cumulative product of exit probability.
    Intended to be used as an argument of Shanks transformation.
    """
    cdef double _prod
    cdef Chain chain

    def __init__(self, chain: Chain):
        self._prod = 1.
        self.chain = chain
        self.reset()

    cdef void reset(self):
        self._prod = 1.

    cdef inline double eval(self, long k):
        cdef double linger_prob = 1 - self.chain.exit_probability_(k)
        if linger_prob >= 1:
            return math.INFINITY
        self._prod *= linger_prob
        return self._prod

    def __call__(self, k: int) -> float:
        return self.eval(k)

cdef unsigned long MAX_N_FINITE_STATES = 2 ** 14
cdef unsigned long MAX_APPROXIMATION_STEPS = 2 ** 10

# noinspection DuplicatedCode
cdef class FiniteChain(Chain):
    """Base class for finite chains
    """
    cpdef double expectation(self, unsigned long max_iter=0):
        cdef double _e = 0., n_states = self.n_states()
        if not max_iter:
            # if n_states is too large, treat it like an infinite chain
            if n_states > MAX_N_FINITE_STATES:
                warn_value(f'Number of states {int(n_states)} is too large, '
                           'using Shanks transformation to approximate.')
                _e = shanks(_ExitProbabilityCumProd(self),
                            start=self._initial_state,
                            max_r=5,
                            max_iter=MAX_APPROXIMATION_STEPS if max_iter == 0 else max_iter)
                if _e < 0:  # large numerical error
                    _e = math.INFINITY
                return 1 + _e
            else:
                max_iter = <unsigned long> n_states + 1
        else:
            max_iter = <unsigned long> math.fmin(n_states + 1, max_iter)
        cdef double _prod = 1., linger_prob
        cdef unsigned long j
        # exact computation up to max_iter
        for j in range(1, max_iter):
            linger_prob = 1 - self.exit_probability_(j + self._initial_state - 1)
            if linger_prob == 0:
                break
            elif linger_prob >= 1:
                _e = math.INFINITY
                break
            _prod *= linger_prob
            _e += _prod
        return 1 + _e

# noinspection DuplicatedCode
cdef class InfiniteChain(Chain):
    """Base class for infinite chains
    """
    cpdef double expectation(self, unsigned long max_iter=500):
        cdef double _e = shanks(_ExitProbabilityCumProd(self),
                                start=self._initial_state,
                                max_r=5,
                                max_iter=MAX_APPROXIMATION_STEPS if max_iter == 0 else max_iter)
        # self._e = float(mpmath.nsum(lambda j: exit_probs_prod(int(j)), [1, mpmath.INFINITY])) + 1
        return 1 + _e

    cpdef double n_states(self):
        return math.INFINITY

cdef class _BuiltinFiniteChain(FiniteChain):
    cpdef double exit_probability(self, unsigned long k):
        return self.exit_probability_(k)

cdef class _BuiltinInfiniteChain(InfiniteChain):
    cpdef double exit_probability(self, unsigned long k):
        return self.exit_probability_(k)

################################################################################
# Finite chains
################################################################################
# noinspection DuplicatedCode
cdef class AbsorbingChain(_BuiltinFiniteChain):
    """A chain that has single absorbing state :math:`p_{\\text{exit}}=0`.
    """
    cdef inline double exit_probability_(self, unsigned long k) nogil:
        return 0

    cpdef double n_states(self):
        return 1

# noinspection DuplicatedCode
cdef class ForwardingChain(_BuiltinFiniteChain):
    """A chain that has :math:`p_{\\text{exit}}=1`.
    """
    cdef inline double exit_probability_(self, unsigned long k) nogil:
        return 1

    cpdef double n_states(self):
        return 1

# noinspection DuplicatedCode
cdef class ListChain(_BuiltinFiniteChain):
    """
    A finite chain with exit probabilities pre-defined and stored in a list.

    Args:
        probs (sequence of float): list of exit (or linger) probabilities.
        is_exit (bool): If True, ``probs`` will contain exit probabilities.
    """
    cdef double[:] exit_probs
    def __init__(self, probs: VectorLike, is_exit: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        # check values
        if (probs < 0).any() or (probs > 1).any():
            raise ValueError('probabilities must be in range [0, 1].')

        if not is_exit:
            probs = 1 - probs
        if not len(probs):
            probs = np.full(1, 1 if is_exit else 0, dtype=np.float64)
        elif probs[-1] != 1:
            probs = np.append(probs, np.array([1.]))
        self.exit_probs = probs

    cdef inline double exit_probability_(self, unsigned long k) nogil:
        return self.exit_probs[k - 1]

    cpdef double n_states(self):
        return self.exit_probs.shape[0]

    def __repr__(self):
        return f'{self.__class__.__name__}(exit_probs={list(self.exit_probs)})'

# noinspection DuplicatedCode
cdef class LinearChain(_BuiltinFiniteChain):
    """
    A finite chain with exit probabilities computed from a linear function:

    .. math::
        p_{\\text{exit}}=c * x

    Args:
        c (float): Non-negative coefficient. If ``c=0``,
         the chain becomes absorbing.
    """
    cdef public double c
    def __init__(self, c: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if c < 0:
            raise ValueError('c must be non-negative. '
                             f'Got c={c}.')
        self.c = c

    cdef inline double exit_probability_(self, unsigned long k) nogil:
        return math.fmin(self.c * k, 1.)

    cpdef double n_states(self):
        return math.ceil(1 / self.c) - self._initial_state + 1 if self.c != 0 else 1

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self.c})'

################################################################################
# Infinite chains
################################################################################
# noinspection DuplicatedCode
cdef class GeometricChain(_BuiltinInfiniteChain):
    """
    Infinite chain with exit probability generated by a Geometric series:

    .. math::
        p_{\\text{exit}}=1 - a \\times r^{k}

    Args:
        a (float): Non-negative coefficient.
        r (float): Common ratio in range (-1, 1).
    """
    cdef public double a, r
    def __init__(self, a: float, r: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if a < 0:
            raise ValueError('a must be non-negative. '
                             f'Got a={a}.')
        if math.fabs(r) >= 1:
            raise ValueError('Series do not converge with |r| >= 1. '
                             f'Got r={r}.')
        self.a = a
        self.r = r

    cdef inline double exit_probability_(self, unsigned long k) nogil:
        return 1 - self.a * self.r ** k

    def __repr__(self):
        return f'{self.__class__.__name__}(a={self.a}, r={self.r})'

# noinspection DuplicatedCode
cdef class HarmonicChain(_BuiltinInfiniteChain):
    """
    Infinite chain with exit probability generated by a
    Harmonic-like series:

    .. math::
        p_{\\text{exit}}=\\frac{c * k}{c * k + 1}

    With ``c=1`` we get the Harmonic series.

    Args:
        c (float): Non-negative number. If ``c=0``,
         the chain becomes absorbing. Defaults to 1.

    Notes:
        Analytical expectation can also be computed as follows:
        ``1 / (c ** (1 / c) * sympy.exp(1 / c) * sympy.lowergamma(1 + 1 / c, 1 / c) + 1)``
    """
    cdef public double c
    def __init__(self, c: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if c < 0:
            raise ValueError('c must be non-negative. '
                             f'Got c={c}.')
        self.c = c

    cdef inline double exit_probability_(self, unsigned long k) nogil:
        cdef double x = self.c * k
        return x / (x + 1)

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self.c})'

# noinspection DuplicatedCode
cdef class TanhChain(_BuiltinInfiniteChain):
    cdef public double c
    def __init__(self, c: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    cdef inline double exit_probability_(self, unsigned long k) nogil:
        return math.tanh(self.c * k)

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self.c})'

# noinspection DuplicatedCode
cdef class SigmoidChain(_BuiltinInfiniteChain):
    cdef public double c
    def __init__(self, c: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    cdef inline double exit_probability_(self, unsigned long k) nogil:
        return 2 / (1 + math.exp(-self.c * k)) - 1

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self.c})'

# noinspection DuplicatedCode
cdef class ATanChain(_BuiltinInfiniteChain):
    cdef public double c
    def __init__(self, c: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    cdef inline double exit_probability_(self, unsigned long k) nogil:
        return 2 / math.pi * math.atan(self.c * k)

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self.c})'

# noinspection DuplicatedCode
cdef class LambdaChain(_BuiltinInfiniteChain):
    """
    Infinite chain with exit probability generated by a custom lambda
    function.

    Args:
        f (callable): Function that takes state as argument and return
         the corresponding exit probability.
    """
    cdef public object f
    def __init__(self, f: Callable[[int], float], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = f

    cdef inline double exit_probability_(self, unsigned long k) nogil:
        cdef double p
        with gil:
            p = self.f(k)
        return p
