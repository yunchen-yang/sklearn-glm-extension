import numpy as np
from sklearn._loss.loss import BaseLoss
from sklearn._loss.link import LogLink
from ._C._loss import CyNegativeBinomialLoss

class HalfNegativeBinomialLoss(BaseLoss):
    """
    Negative Binomial loss function wrapper.
    """
    def __init__(self, sample_weight=None, k=1.0):
        if k < 0:
            raise ValueError("Dispersion parameter k must be non-negative.")
        super().__init__(closs=CyNegativeBinomialLoss(k=k), link=LogLink(), n_classes=1)
