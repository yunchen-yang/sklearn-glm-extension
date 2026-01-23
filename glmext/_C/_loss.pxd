ctypedef fused floating_in:
    double
    float


ctypedef fused floating_out:
    double
    float


ctypedef struct double_pair:
    double val1
    double val2


cdef class CyNegativeBinomialLoss:
    cdef readonly double k
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
