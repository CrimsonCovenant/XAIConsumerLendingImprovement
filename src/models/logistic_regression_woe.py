"""From-scratch logistic regression with Adam optimizer and cosine LR decay.
This is the project's baseline model. It is implemented entirely in NumPy
(no sklearn) so that every gradient step is auditable and transparent.
Key design decisions:
  1. Adam optimizer instead of plain SGD -- eliminates oscillation that
     kept the naive gradient descent stuck near the random baseline.
  2. Class weights so approved and denied applications contribute equally
     to the loss, despite the ~75/25 imbalance in HMDA data.
  3. L2 regularization on weights only (bias excluded) to prevent
     overfitting without biasing the decision boundary.
  4. Cosine LR decay so the learning rate starts high for fast convergence
     and smoothly anneals to near-zero for stable final weights.
"""
import numpy as np

class LogisticRegressionAdam:
    """From-scratch logistic regression with Adam, cosine LR, L2, and class weights."""
    def __init__(
        self,
        lr: float = 0.001,
        n_epochs: int = 3000,
        batch_size: int | None = 512,
        lambda_l2: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        verbose: bool = False,
    ):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda_l2 = lambda_l2
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.verbose = verbose
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _class_weights(y: np.ndarray) -> np.ndarray:
        """Compute per-sample weights so each class contributes equally.
        Without this, the model would be biased toward predicting 'approved'
        because approvals make up ~75% of the data.
        """
        n = len(y)
        n_pos = max(y.sum(), 1)
        n_neg = max(n - n_pos, 1)
        return np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))

    def _cosine_lr(self, epoch: int) -> float:
        """Cosine annealing: starts at peak lr, decays to near-zero."""
        return self.lr * (1.0 + np.cos(np.pi * epoch / self.n_epochs)) / 2.0

    def _compute_loss(self, X: np.ndarray, y: np.ndarray, sw: np.ndarray) -> float:
        """Weighted negative log-likelihood plus L2 penalty on weights."""
        eps = 1e-9
        p = self._sigmoid(X @ self.weights + self.bias)
        nll = -np.mean(
            sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        )
        l2 = (self.lambda_l2 / (2 * len(y))) * np.sum(self.weights ** 2)
        return float(nll + l2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionAdam":
        """Train the model using mini-batch Adam with cosine LR decay.
        X: feature matrix (n_samples, n_features), already WOE-encoded.
        y: binary labels (0=denied, 1=approved).
        """
        n, f = X.shape
        self.weights = np.zeros(f)
        self.bias = 0.0
        y = y.ravel()
        sw = self._class_weights(y)
        # Adam moment accumulators (first and second moment estimates)
        mw = np.zeros(f); vw = np.zeros(f)
        mb = 0.0; vb = 0.0
        t = 0  # global step counter for Adam bias correction
        batch_size = self.batch_size if self.batch_size else n
        for epoch in range(self.n_epochs):
            lr_t = self._cosine_lr(epoch)
            # Shuffle data each epoch for stochastic mini-batching
            perm = np.random.permutation(n)
            X_s, y_s, sw_s = X[perm], y[perm], sw[perm]
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb, yb, wb = X_s[start:end], y_s[start:end], sw_s[start:end]
                nb = len(yb)
                t += 1
                p = self._sigmoid(Xb @ self.weights + self.bias)
                residual = (p - yb) * wb
                # Gradient: data term + L2 on weights only (not bias)
                dw = (1.0 / nb) * Xb.T @ residual + (self.lambda_l2 / nb) * self.weights
                db = (1.0 / nb) * residual.sum()
                # Adam update: track running averages of gradient and squared gradient
                mw = self.beta1 * mw + (1 - self.beta1) * dw
                mb = self.beta1 * mb + (1 - self.beta1) * db
                vw = self.beta2 * vw + (1 - self.beta2) * dw ** 2
                vb = self.beta2 * vb + (1 - self.beta2) * db ** 2
                # Bias-corrected estimates (important in early steps)
                mw_h = mw / (1 - self.beta1 ** t)
                mb_h = mb / (1 - self.beta1 ** t)
                vw_h = vw / (1 - self.beta2 ** t)
                vb_h = vb / (1 - self.beta2 ** t)
                self.weights -= lr_t * mw_h / (np.sqrt(vw_h) + self.eps)
                self.bias    -= lr_t * mb_h / (np.sqrt(vb_h) + self.eps)
            if self.verbose and epoch % 300 == 0:
                loss = self._compute_loss(X, y, sw)
                self.loss_history.append(loss)
                print(f"  Epoch {epoch:4d}/{self.n_epochs}  loss={loss:.5f}  lr={lr_t:.6f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(approved | X) as a 1-D array."""
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self) -> tuple[np.ndarray, float]:
        """Return (weights, bias) for inspection or serialization."""
        return self.weights, self.bias
