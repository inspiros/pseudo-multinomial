# distutils: language = c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Sequence, Optional, Union

import numpy as np
cimport numpy as np
from libc cimport math

from ._types import VectorLike, MatrixLike
from .chains import Chain
from .chains cimport Chain

__all__ = [
    'PseudoMultinomialGenerator',
    'pseudo_multinomial',
]

# noinspection DuplicatedCode
cdef class PseudoMultinomialGenerator:
    r"""
    Pseudo-Multinomial Generator class based on Markov chains.
    """
    # cdef readonly list chains
    cdef readonly np.ndarray chains
    cdef void** _chains_ptr
    cdef readonly unsigned long n_chains
    cdef public np.ndarray S
    cdef double[:, :] _S_cumsum
    cdef public object random_state

    cdef public unsigned long _chain_id, _chain_state

    def __init__(self,
                 chains: Sequence[Chain],
                 chain_transition_matrix: MatrixLike,
                 random_state: Optional[Union[np.random.RandomState, int]] = None):
        if not len(chains):
            raise ValueError('No chain.')
        chain_transition_matrix = np.asarray(chain_transition_matrix, dtype=np.float64)
        if (chain_transition_matrix.ndim != 2 or
                chain_transition_matrix.shape[0] != chain_transition_matrix.shape[1]):
            raise ValueError('Invalid chain transition matrix.')
        if len(chains) != chain_transition_matrix.shape[0]:
            raise ValueError('Chain transition matrix does not '
                             'match number of chains.')
        if not isinstance(random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        self.chains = np.asarray(chains)
        self._chains_ptr = <void **> self.chains.data
        self.n_chains = len(self.chains)
        self.S = chain_transition_matrix
        self._S_cumsum = np.cumsum(self.S, axis=1)

        self._chain_id = 0
        self._chain_state = self.chains[self._chain_id]._initial_state

    def set_random_state(self, random_state: Optional[Union[np.random.RandomState, int]] = None) -> None:
        if not isinstance(random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

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
        r"""
        Set a random initial state based on stationary distribution.

        Args:
            chain_id (int, optional): Initial chain. Choose a random chain
             if set to negative value. Defaults to -1.
            max_chain_state (int, optional): Maximum state of chain to
             avoid indefinite loop in infinite chains. Non-positive value
             will disable this. Defaults to 1000.
        """
        cdef double p = self.random_state.rand()
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
            stationary *= 1.0 - (<Chain> self._chains_ptr[chain_id]).exit_probability(i)
            stationary_cumsum += stationary
            i += 1
            if i >= max_chain_state > 0:
                break
        self.set_state(chain_id, i)

    cdef inline unsigned long next_state(self):
        cdef double p = self.random_state.rand()
        self._chain_state = (<Chain> self._chains_ptr[self._chain_id]).next_state(p)

        cdef unsigned long i
        if not self._chain_state:  # 0 = exit
            p = self.random_state.rand()
            for i in range(self.n_chains):
                if p < self._S_cumsum[self._chain_id, i]:
                    self._chain_id = i
                    break
            (<Chain> self._chains_ptr[self._chain_id]).reset()
            self._chain_state = (<Chain> self._chains_ptr[self._chain_id])._state
        return self._chain_id

    cpdef np.ndarray[np.int64_t, ndim=1] next_states(self, unsigned long n = 1):
        r"""
        Generate ``n`` next states.

        Args:
            n (int): Number of states. Defaults to 1.

        Returns:
            states (np.ndarray): Array containing ``n`` next states.
        """
        cdef np.ndarray[np.int64_t, ndim=1] states = np.empty(n, dtype=np.int64)
        cdef np.int64_t[:] states_view = states
        cdef unsigned long i
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

    def generate(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        r"""
        Draw samples.

        Args:
            size (int, optional): Output shape. Default to None, in which case a
            single value is returned.

        Returns:
            out (int or ndarray): `size`-shaped array of random integers,
            or a single such random int if `size` not provided.
        """
        if size is None:
            return self.next_state()
        if size < 1:
            raise ValueError(f'size must be positive. Got {size}.')
        return self.next_states(size)

    @staticmethod
    def from_pvals(chains: Sequence[Chain],
                   pvals: Optional[VectorLike] = None,
                   repeat: Union[bool, Sequence[bool], np.ndarray] = True,
                   random_state: Optional[Union[np.random.RandomState, int]] = None) -> 'PseudoMultinomialGenerator':
        cdef unsigned long n_chains = len(chains)
        if pvals is None:
            pvals = np.ones(n_chains, dtype=np.float64) / n_chains
        elif isinstance(pvals, (np.ndarray, Sequence)):
            pvals = np.asarray(pvals, dtype=np.float64)
        else:
            raise ValueError(f'pvals must be an array of floats. Got {pvals}.')
        if pvals.ndim != 1 or pvals.shape[0] != n_chains:
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
        return PseudoMultinomialGenerator(chains, S, random_state)

def pseudo_multinomial(chain: Union[PseudoMultinomialGenerator, Sequence[Chain], Chain],
                       size: Optional[int] = None,
                       random_init: bool = True,
                       **kwargs) -> Union[int, np.ndarray]:
    r"""
    Draw samples from the pseudo-random multinomial distribution.

    Args:
        chain (PseudoMultinomialGenerator or sequence of Chain): An instance
            of :class:`PseudoMultinomialGenerator`, or a sequence of
            :class:`Chain`s for initializing one.
        size (int, optional): Output shape. Default to None, in which case a
            single value is returned.
        random_init (bool): Reset the Markov chain at a random initial state.
        **kwargs: Extra keyword arguments passed to ``from_pvals``.

    Returns:
        out (int or ndarray): `size`-shaped array of random integers,
            or a single such random int if `size` not provided.
    """
    if isinstance(chain, Sequence):
        chain = PseudoMultinomialGenerator.from_pvals(chain, **kwargs)
    elif isinstance(chain, Chain):
        chain = PseudoMultinomialGenerator.from_pvals([chain])
    elif not isinstance(chain, PseudoMultinomialGenerator):
        raise ValueError('chain must be a PseudoMultinomialGenerator or '
                         f'a sequence of Chain. Got {type(chain)}.')

    if random_init:
        chain.random_init()
    return chain.generate(size)
