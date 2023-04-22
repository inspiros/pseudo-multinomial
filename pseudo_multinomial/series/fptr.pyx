# distutils: language = c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

__all__ = ['DoubleSeriesFPtr', 'CyDoubleSeriesFPtr', 'PyDoubleSeriesFPtr']

# --------------------------------
# Series evaluating functions
# --------------------------------
ctypedef double (*sf_ptr)(long)

cdef class TrackedFPtr:
    def __cinit__(self):
        self.n_f_calls = 0

cdef class DoubleSeriesFPtr(TrackedFPtr):
    def __call__(self, long k):
        return self.eval(k)

    cdef double eval(self, long k) except*:
        raise NotImplementedError

cdef class CyDoubleSeriesFPtr(DoubleSeriesFPtr):
    def __init__(self):
        raise TypeError('This class cannot be instantiated directly.')

    @staticmethod
    cdef CyDoubleSeriesFPtr from_f(sf_ptr f):
        cdef CyDoubleSeriesFPtr wrapper = CyDoubleSeriesFPtr.__new__(CyDoubleSeriesFPtr)
        wrapper.f = f
        return wrapper

    cdef inline double eval(self, long k) except*:
        self.n_f_calls += 1
        return self.f(k)

cdef class PyDoubleSeriesFPtr(DoubleSeriesFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef DoubleSeriesFPtr from_f(object f):
        if isinstance(f, DoubleSeriesFPtr):
            return f
        cdef PyDoubleSeriesFPtr wrapper = PyDoubleSeriesFPtr.__new__(PyDoubleSeriesFPtr)
        wrapper.f = f
        return wrapper

    cdef inline double eval(self, long k) except*:
        self.n_f_calls += 1
        return self.f(k)
