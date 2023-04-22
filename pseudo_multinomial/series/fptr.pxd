cdef class TrackedFPtr:
    cdef public unsigned long n_f_calls

# --------------------------------
# Double
# --------------------------------
ctypedef double (*dsf_ptr)(long)

cdef class DoubleSeriesFPtr(TrackedFPtr):
    cdef double eval(self, long k) except *

cdef class CyDoubleSeriesFPtr(DoubleSeriesFPtr):
    cdef dsf_ptr f
    @staticmethod
    cdef CyDoubleSeriesFPtr from_f(dsf_ptr f)
    cdef double eval(self, long k) except *

cdef class PyDoubleSeriesFPtr(DoubleSeriesFPtr):
    cdef object f
    @staticmethod
    cdef DoubleSeriesFPtr from_f(object f)
    cdef double eval(self, long k) except *

ctypedef fused double_series_func_type:
    dsf_ptr
    DoubleSeriesFPtr
