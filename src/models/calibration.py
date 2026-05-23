"""Platt scaling (2-parameter sigmoid calibration).
When a model says '70% chance of approval', Platt scaling adjusts that
number so it actually matches reality. It fits two parameters (a, b) on
a held-out calibration slice by minimizing negative log-likelihood:
    P_calibrated = sigmoid(a * logit(P_raw) + b)
The calibration set must be separate from both training and test sets.
We use a 10% slice of the full dataset for this purpose.
"""
from __future__ import annotations
import numpy as np

class PlattScaler:
    """Two-parameter Platt scaling for probability calibration.
    Learns a and b by gradient descent on a calibration set, then
    applies the learned transform to produce well-calibrated probabilities.
    """
    def __init__(self, lr: float = 0.01, n_iters: int = 5000):
        self.lr = lr
        self.n_iters = n_iters
        self.a: float = 1.0
        self.b: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        # Clip to avoid log(0) or log(negative)
        p_clipped = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p_clipped / (1 - p_clipped))

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattScaler":
        """Learn a and b from calibration data.
        p_raw: raw predicted probabilities from the model.
        y: true binary labels (0 = denied, 1 = approved).
        """
        p_raw = np.asarray(p_raw).ravel()
        y = np.asarray(y).ravel()
        logits = self._logit(p_raw)
        self.a = 1.0
        self.b = 0.0
        # Simple gradient descent on negative log-likelihood.
        # We use plain GD instead of Adam here because the 2-parameter
        # surface is nearly convex and converges reliably in 5000 steps.
        for _ in range(self.n_iters):
            z = self.a * logits + self.b
            p_cal = self._sigmoid(z)
            # Gradient of NLL: residual = predicted - actual
            residual = p_cal - y
            da = np.mean(residual * logits)
            db = np.mean(residual)
            self.a -= self.lr * da
            self.b -= self.lr * db
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        """Apply the learned calibration to new raw probabilities."""
        logits = self._logit(np.asarray(p_raw).ravel())
        return self._sigmoid(self.a * logits + self.b)
