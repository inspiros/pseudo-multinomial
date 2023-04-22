cimport numpy as np

cdef class Chain:
    cdef readonly unsigned long _initial_state, _state
    cdef void reset(self) nogil
    cdef unsigned long state(self) nogil
    cdef void set_state(self, unsigned long k) nogil
    cdef unsigned long next_state(self, double p) nogil
    cdef double exit_probability_(self, unsigned long k) nogil
    cpdef double exit_probability(self, unsigned long k)
    cdef double linger_probability_(self, unsigned long k) nogil
    cpdef double expectation(self)
    cpdef double n_states(self)
    cpdef bint is_finite(self)
    cpdef np.ndarray[np.float64_t, ndim=2] transition_matrix(self, unsigned long n=*)
