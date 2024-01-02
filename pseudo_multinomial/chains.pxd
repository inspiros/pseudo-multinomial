cimport numpy as np

cdef class Chain:
    cdef readonly unsigned long _initial_state, _state
    cdef void reset(self)
    cdef unsigned long state(self)
    cdef void set_state(self, unsigned long k)
    cdef unsigned long next_state(self, double p)
    cpdef double exit_probability(self, unsigned long k)
    cpdef double linger_probability(self, unsigned long k)
    cpdef double expectation(self)
    cpdef double n_states(self)
    cpdef bint is_finite(self)
    cpdef np.ndarray[np.float64_t, ndim=2] transition_matrix(self, unsigned long n=*)
