import numpy as np

class LogisticRegressionOptimized:
    """
    An optimized Logistic Regression model using Mini-Batch Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=512, verbose=False):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs # Renamed from n_iterations for clarity
        self.batch_size = batch_size
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Private method to compute the sigmoid function."""
        # Clip z to avoid overflow in np.exp
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def fit(self, X, y):
        """
        Trains the logistic regression model using Mini-Batch Gradient Descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        y = y.reshape(-1, 1) # Ensure y is a column vector
        
        # Mini-Batch Gradient Descent
        for epoch in range(self.n_epochs):
            # Shuffle the data at the beginning of each epoch
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                n_batch_samples = X_batch.shape[0]

                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_predicted = self._sigmoid(linear_model).reshape(-1, 1)

                # Compute gradients on the mini-batch
                dw = (1 / n_batch_samples) * np.dot(X_batch.T, (y_predicted - y_batch)).flatten()
                db = (1 / n_batch_samples) * np.sum(y_predicted - y_batch)

                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            if self.verbose and (epoch % 10 == 0):
                cost = self._compute_cost(X, y.flatten())
                print(f"Epoch {epoch}: Cost = {cost:.4f}")

    def predict_proba(self, X):
        """Returns the probability estimates for the positive class."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Predicts the class label for a given input."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
        
    def _compute_cost(self, X, y):
        """Private method to compute the NLL (cross-entropy) cost."""
        n_samples = len(y)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        epsilon = 1e-9 # for numerical stability
        cost = (-1/n_samples) * np.sum(y * np.log(y_predicted + epsilon) + (1-y) * np.log(1 - y_predicted + epsilon))
        return cost

    def get_params(self):
        """Returns the learned weights and bias."""
        return self.weights, self.bias