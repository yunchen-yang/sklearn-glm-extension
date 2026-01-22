# _loss.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport log, exp
from cython.parallel import prange


cdef inline double closs_negative_binomial(
    double y_true,
    double raw_prediction,
    double k
) noexcept nogil:
    """
    The loss function (Negative Log-Likelihood) of the Negative Binomial GLM for a single sample.
    """
    if k == 0:
        return exp(raw_prediction) - y_true * raw_prediction
    else:
        inv_k = 1.0 / k
        return (y_true + inv_k) * log(1.0 + k * exp(raw_prediction)) - y_true * raw_prediction


cdef inline double cgrad_negative_binomial(
    double y_true,
    double raw_prediction,
    double k
) noexcept nogil:
    if k == 0:
        return exp(raw_prediction) - y_true
    else:
        return (exp(raw_prediction) - y_true) / (1.0 + k * exp(raw_prediction))


cdef inline double_pair closs_grad_negative_binomial(
    double y_true,
    double raw_prediction,
    double k
) noexcept nogil:
    cdef double_pair lg
    if k == 0:
        lg.val1 = exp(raw_prediction) - y_true * raw_prediction
        lg.val2 = exp(raw_prediction) - y_true
    else:
        inv_k = 1.0 / k
        lg.val1 = (y_true + inv_k) * log(1.0 + k * exp(raw_prediction)) - y_true * raw_prediction
        lg.val2 = (exp(raw_prediction) - y_true) / (1.0 + k * exp(raw_prediction))
    return lg


cdef class CyNegativeBinomialLoss:
    def __init__(self, k):
        self.k = k
    
    def __reduce__(self):
        return (self.__class__, (self.k,))
    
    cdef inline double cy_loss(self, double y_true, double raw_prediction) noexcept nogil:
        return closs_negative_binomial(y_true, raw_prediction, self.k)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil:
        return cgrad_negative_binomial(y_true, raw_prediction, self.k)

    #cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil:
        #return {{cgrad_hess}}(y_true, raw_prediction{{with_param}})
    
    def loss(
        self,
        const floating_in[::1] y_true,          # IN
        const floating_in[::1] raw_prediction,  # IN
        const floating_in[::1] sample_weight,   # IN
        floating_out[::1] loss_out,             # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_negative_binomial(y_true[i], raw_prediction[i], self.k)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_negative_binomial(y_true[i], raw_prediction[i], self.k)

    def loss_gradient(
        self,
        const floating_in[::1] y_true,          # IN
        const floating_in[::1] raw_prediction,  # IN
        const floating_in[::1] sample_weight,   # IN
        floating_out[::1] loss_out,             # OUT
        floating_out[::1] gradient_out,         # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_negative_binomial(y_true[i], raw_prediction[i], self.k)
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_negative_binomial(y_true[i], raw_prediction[i], self.k)
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

    def gradient(
        self,
        const floating_in[::1] y_true,          # IN
        const floating_in[::1] raw_prediction,  # IN
        const floating_in[::1] sample_weight,   # IN
        floating_out[::1] gradient_out,         # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgrad_negative_binomial(y_true[i], raw_prediction[i], self.k)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgrad_negative_binomial(y_true[i], raw_prediction[i], self.k)