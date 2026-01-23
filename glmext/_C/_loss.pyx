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
    cdef double mu = exp(raw_prediction)
    if k == 0:
        return mu - y_true
    else:
        return (mu - y_true) / (1.0 + k * mu)


cdef inline double_pair closs_grad_negative_binomial(
    double y_true,
    double raw_prediction,
    double k
) noexcept nogil:
    cdef double_pair lg
    cdef double mu = exp(raw_prediction)
    if k == 0:
        lg.val1 = mu - y_true * raw_prediction
        lg.val2 = mu - y_true
    else:
        inv_k = 1.0 / k
        lg.val1 = (y_true + inv_k) * log(1.0 + k * mu) - y_true * raw_prediction
        lg.val2 = (mu - y_true) / (1.0 + k * mu)
    return lg


cdef inline double_pair cgrad_hess_negative_binomial(
    double y_true,
    double raw_prediction,
    double k
) noexcept nogil:
    """
    The Hessian (2nd derivative) of the Negative Binomial GLM
    with respect to the raw_prediction (log-link).
    """
    cdef double_pair gh
    cdef double mu = exp(raw_prediction)
    cdef double denom = 1.0 + k * mu
    if k == 0:
        gh.val1 = mu - y_true
        # Poisson limit: H = mu
        gh.val2 = mu
    else:
        gh.val1 = (mu - y_true) / (1.0 + k * mu)
        # Formula: H = mu * (1 + k * y) / (1 + k * mu)^2
        gh.val2 = mu * (1.0 + k * y_true) / (denom * denom)
    return gh


cdef class CyNegativeBinomialLoss:
    def __init__(self, k):
        self.k = k
    
    def __reduce__(self):
        return (self.__class__, (self.k,))
    
    cdef inline double cy_loss(self, double y_true, double raw_prediction) noexcept nogil:
        return closs_negative_binomial(y_true, raw_prediction, self.k)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil:
        return cgrad_negative_binomial(y_true, raw_prediction, self.k)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil:
        return cgrad_hess_negative_binomial(y_true, raw_prediction, self.k)
    
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
    
    def gradient_hessian(
        self,
        const floating_in[::1] y_true,          # IN
        const floating_in[::1] raw_prediction,  # IN
        const floating_in[::1] sample_weight,   # IN
        floating_out[::1] gradient_out,         # OUT
        floating_out[::1] hessian_out,          # OUT
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
                dbl2 = cgrad_hess_negative_binomial(y_true[i], raw_prediction[i], self.k)
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_negative_binomial(y_true[i], raw_prediction[i], self.k)
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2
