# distutils: language = c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

import numpy as np
cimport numpy as np
from libc cimport math

from .chains import Chain
from .chains cimport Chain
from .typing import Sequence, Optional, Union, VectorLike, MatrixLike
from .utils.random_utils cimport mt19937, seed_mt19937, uniform_real_distribution

# cdef extern from "<random>" namespace "std":
#     cdef cppclass pcg64:
#         pcg64() nogil
#         pcg64(unsigned int seed) nogil
#
#     cdef cppclass mt19937:
#         mt19937() nogil
#         mt19937(unsigned int seed) nogil
#
#     cdef cppclass uniform_real_distribution[double]:
#         uniform_real_distribution() nogil
#         uniform_real_distribution(double a, double b) nogil
#         double operator()(pcg64 gen) nogil
#         double operator()(mt19937 gen) nogil

# cdef extern from "./utils/random_utils.h":
#     cdef void seed_mt19937(mt19937 gen) nogil
#     cdef void seed_mt19937(mt19937 gen, int seed) nogil

__all__ = [
    'MasterChain',
    'pseudo_multinomial',
]

# noinspection DuplicatedCode
cdef class MasterChain:
    # cdef readonly list chains
    cdef readonly np.ndarray chains
    cdef void** _chains_ptr
    cdef readonly unsigned long n_chains
    cdef public np.ndarray S
    cdef double[:, :] _S_cumsum
    cdef mt19937 _rng
    cdef uniform_real_distribution[double] _dist

    cdef public unsigned long _chain_id, _chain_state

    def __init__(self,
                 chains: Sequence[Chain],
                 chain_transition_matrix: MatrixLike,
                 seed: Optional[int] = None):
        if not len(chains):
            raise ValueError('No chain.')
        chain_transition_matrix = np.asarray(chain_transition_matrix, dtype=np.float64)
        if (chain_transition_matrix.ndim != 2 or
                chain_transition_matrix.shape[0] != chain_transition_matrix.shape[1]):
            raise ValueError('Invalid chain transition matrix.')
        if len(chains) != chain_transition_matrix.shape[0]:
            raise ValueError('Chain transition matrix does not '
                             'match number of chains.')
        self.chains = np.asarray(chains)
        self._chains_ptr = <void **> self.chains.data
        self.n_chains = len(self.chains)
        self.S = chain_transition_matrix
        self._S_cumsum = np.cumsum(self.S, axis=1)

        self._rng = mt19937()
        self._dist = uniform_real_distribution()
        self.set_seed(seed)

        self._chain_id = 0
        self._chain_state = self.chains[self._chain_id]._initial_state

    def set_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed_mt19937(self._rng)
        else:
            seed_mt19937(self._rng, <int> seed)

    cdef void set_mt19937(self, mt19937 _mt19937) nogil:
        self._rng = _mt19937

    cpdef (unsigned long, unsigned long) state(self):
        return self._chain_id, self._chain_state

    cpdef void set_state(self, unsigned long chain_id, unsigned long chain_state):
        if chain_id >= self.n_chains:
            raise KeyError('state out of bound.')
        if chain_state > (<Chain> self._chains_ptr[chain_id]).n_states():
            raise KeyError('chain_state out of bound.')
        (<Chain> self._chains_ptr[self._chain_id]).reset()
        self._chain_id = chain_id
        self._chain_state = (chain_state if chain_state > 0
                             else (<Chain> self._chains_ptr[self._chain_id])._initial_state)
        (<Chain> self._chains_ptr[self._chain_id]).set_state(self._chain_state)

    cpdef void random_init(self, long chain_id = -1, long max_chain_state = 1000):
        """
        Set a random initial state based on stationary distribution.

        Args:
            chain_id (int, optional): Initial chain. Choose a random chain
             if set to negative value. Defaults to -1.
            max_chain_state (int, optional): Maximum state of chain to
             avoid indefinite loop in infinite chains. Non-positive value
             will disable this. Defaults to 1000.
        """
        cdef double p = self._dist(self._rng)
        cdef np.ndarray[np.float64_t, ndim=2] es = self.entrance_stationaries()
        cdef np.ndarray[np.float64_t, ndim=1] probs = es[0] * es[1]
        cdef np.ndarray[np.float64_t, ndim=1] probs_cumsum = np.empty(self.n_chains + 1, dtype=np.float64)
        probs_cumsum[0] = 0
        probs_cumsum[1:] = np.cumsum(probs)
        cdef unsigned long i
        if chain_id >= 0 and chain_id >= self.n_chains:
            raise KeyError(f'chain_id={chain_id} out of bound.')
        else:
            for i in range(self.n_chains):
                if p < probs_cumsum[i + 1]:
                    chain_id = i
                    p = (p - probs_cumsum[i]) / (probs_cumsum[i + 1] - probs_cumsum[i])
                    break
        i = (<Chain> self._chains_ptr[chain_id])._initial_state
        cdef double stationary = 1 / es[1, chain_id]
        cdef double stationary_cumsum = stationary
        while p > stationary_cumsum:
            stationary *= (<Chain> self._chains_ptr[chain_id]).linger_probability_(i)
            stationary_cumsum += stationary
            i += 1
            if i >= max_chain_state > 0:
                break
        self.set_state(chain_id, i)

    cdef inline unsigned long next_state(self) nogil:
        cdef double p = self._dist(self._rng)
        self._chain_state = (<Chain> self._chains_ptr[self._chain_id]).next_state(p)

        cdef unsigned long i
        if not self._chain_state:  # 0 = exit
            p = self._dist(self._rng)
            for i in range(self.n_chains):
                if p < self._S_cumsum[self._chain_id, i]:
                    self._chain_id = i
                    break
            (<Chain> self._chains_ptr[self._chain_id]).reset()
            self._chain_state = (<Chain> self._chains_ptr[self._chain_id])._state
        return self._chain_id

    cpdef np.ndarray[np.int64_t, ndim=1] next_states(self, unsigned long n = 1):
        """
        Generate ``n`` next states.

        Args:
            n (int): Number of states. Defaults to 1.

        Returns:
            states (np.ndarray): Array containing ``n`` next states.
        """
        cdef np.ndarray[np.int64_t, ndim=1] states = np.empty(n, dtype=np.int64)
        cdef np.int64_t[:] states_view = states
        cdef unsigned long i
        with nogil:
            for i in range(n):
                states_view[i] = self.next_state()
        return states

    cpdef np.ndarray[np.float64_t, ndim=1] n_states(self):
        cdef np.ndarray[np.float64_t, ndim=1] ns = np.empty(self.n_chains, dtype=np.float64)
        cdef unsigned long i
        for i in range(self.n_chains):
            ns[i] = (<Chain> self._chains_ptr[i]).n_states()
        return ns

    cpdef np.ndarray[np.float64_t, ndim=1] expectations(self):
        cdef np.ndarray[np.float64_t, ndim=1] e = np.empty(self.n_chains, dtype=np.float64)
        for i in range(self.n_chains):
            e[i] = (<Chain> self._chains_ptr[i]).expectation()
        return e / (1 - self.S.diagonal())

    cpdef np.ndarray[np.float64_t, ndim=2] entrance_stationaries(self):
        cdef np.ndarray[np.float64_t, ndim=2] A = \
            np.empty((self.n_chains + 1, self.n_chains), dtype=np.float64)
        A[:-1] = self.S.T
        cdef unsigned long i
        for i in range(self.n_chains):
            A[i, i] -= 1
        for i in range(self.n_chains):
            A[-1, i] = (<Chain> self._chains_ptr[i]).expectation()
        cdef np.ndarray[np.float64_t, ndim=1] b = \
            np.zeros(self.n_chains + 1, dtype=np.float64)
        b[-1] = 1
        cdef np.ndarray[np.float64_t, ndim=2] es = np.empty((2, self.n_chains), dtype=np.float64)
        if not np.isinf(A[-1]).any():
            es[0] = np.linalg.lstsq(A, b, rcond=None)[0]
            # es[0] = np.linalg.pinv(A).dot(b)
        else:
            for i in range(self.n_chains):
                es[0, i] = 1 if math.isinf(A[-1, i]) else 0
            es[0] /= es[0].sum()
        es[1] = A[-1]
        return es

    cpdef np.ndarray[np.float64_t, ndim=1] probs(self):
        cdef np.ndarray[np.float64_t, ndim=2] es = self.entrance_stationaries()
        return es[0] * es[1]

    cpdef bint is_finite(self):
        cdef unsigned long i
        for i in range(self.n_chains):
            if not (<Chain> self._chains_ptr[i]).is_finite():
                return False
        return True

    cpdef np.ndarray[np.float64_t, ndim=2] chain_transition_matrix(self):
        return self.S

    cpdef np.ndarray[np.float64_t, ndim=2] transition_matrix(self, unsigned long n=0):
        if not self.is_finite() and n == 0:
            raise ValueError('Complete transition matrix is available only '
                             'when all chains are finite.')
        cdef np.ndarray[np.uint64_t, ndim=1] n_states = \
            np.empty(self.n_chains, dtype=np.uint64)
        cdef unsigned long i
        for i in range(self.n_chains):
            n_states[i] = <np.uint64_t> (min((<Chain> self._chains_ptr[i]).n_states(), n)
                                         if n > 0 else (<Chain> self._chains_ptr[i]).n_states())
        cdef np.ndarray[np.uint64_t, ndim=1] initial_dims = \
            np.empty(self.n_chains + 1, dtype=np.uint64)
        initial_dims[0] = 0
        initial_dims[1:] = np.cumsum(n_states)

        cdef unsigned long total_n_states = n_states.sum()
        cdef np.ndarray[np.float64_t, ndim=2] P_i, P = \
            np.zeros((total_n_states, total_n_states), dtype=np.float64)
        cdef unsigned long start_dim, end_dim
        for i in range(self.n_chains):
            start_dim = initial_dims[i]
            end_dim = initial_dims[i + 1]
            P_i = (<Chain> self._chains_ptr[i]).transition_matrix(n_states[i])
            P[start_dim:end_dim, start_dim:end_dim] = P_i[1:, 1:]
            P[start_dim:end_dim, initial_dims[:-1]] = np.outer(1 - P_i[1:, 1:].sum(1), self.S[i])
        return P

    def __len__(self) -> int:
        return self.chains.shape[0]

    def __next__(self) -> int:
        cdef unsigned long next_state = self.next_state()
        return next_state

    def __getitem__(self, item) -> Chain:
        return self.chains[item]

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}('
        for i in range(self.n_chains):
            repr_str += f'\n\t{i}: {repr(self.chains[i])},'
        repr_str += '\n)'
        return repr_str

    def pseudo_multinomial(self, n: int = 1, random_init: bool = True):
        if n < 1:
            raise ValueError(f'n must be positive. Got {n}.')
        if random_init:
            self.random_init()
        cdef np.ndarray[np.int64_t, ndim=1] rands = self.next_states(n)
        if n == 1:
            return rands.item()
        return rands

    @staticmethod
    def from_pvals(chains: Sequence[Chain],
                   pvals: Optional[Union[VectorLike, float]] = None,
                   repeat: Union[bool, Sequence[bool], np.ndarray] = True,
                   random_state: Optional[np.random.RandomState] = None) -> 'MasterChain':
        cdef unsigned long n_chains = len(chains)
        if pvals is None:
            pvals = np.ones(n_chains, dtype=np.float64) / n_chains
        elif isinstance(pvals, float):
            pvals = np.full(n_chains, pvals, dtype=np.float64)
        else:
            pvals = np.asarray(pvals, dtype=np.float64)
        if pvals.shape[0] != n_chains:
            raise ValueError('chains and pvals must have the same number '
                             f'of elements. Got {n_chains}, {pvals.shape[0]}.')
        if pvals[:-1].sum() > 1:
            raise ValueError('sum(pvals[:-1]) > 1.0')
        # set last element of pvals as the complement of the rest
        pvals[-1] = 1 - pvals[:-1].sum()

        if not isinstance(repeat, (np.ndarray, Sequence)):
            repeat = [repeat]
        repeat = np.asarray(repeat, dtype=np.bool_)
        if repeat.shape[0] == 1:
            repeat = np.repeat(repeat, pvals.shape[0])
        elif repeat.shape[0] != pvals.shape[0]:
            raise ValueError('chains and repeat must have the same number '
                             f'of elements. Got {n_chains}, {repeat.shape[0]}')

        cdef np.ndarray[np.float64_t, ndim=2] S = \
            np.empty((n_chains, n_chains), dtype=np.float64)
        cdef unsigned long i
        for i in range(n_chains):
            S[:, i] = pvals[i]
            if not repeat[i]:
                S[i, i] = 0
        S /= S.sum(1, keepdims=True)
        return MasterChain(chains, S, random_state)

def pseudo_multinomial(chain: Union[MasterChain, Sequence[Chain], Chain],
                       n: int = 1,
                       random_init: bool = True,
                       **kwargs):
    if isinstance(chain, Sequence):
        chain = MasterChain.from_pvals(chain, **kwargs)
    elif isinstance(chain, Chain):
        chain = MasterChain.from_pvals([chain])
    else:
        raise ValueError('chain is not a chain?!')
    return chain.pseudo_multinomial(n, random_init)
