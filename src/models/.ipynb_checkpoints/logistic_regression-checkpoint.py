import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Private method to compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Trains the logistic regression model using gradient descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients based on the NLL loss function
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if self.verbose and (i % 100 == 0):
                cost = self._compute_cost(X, y)
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict_proba(self, X):
        """
        Returns the probability estimates for the positive class.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predicts the class label for a given input.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
        
    def _compute_cost(self, X, y):
        """Private method to compute the NLL cost."""
        n_samples = len(y)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        # Add a small epsilon for numerical stability in log
        epsilon = 1e-9
        cost = (-1/n_samples) * np.sum(y * np.log(y_predicted + epsilon) + (1-y) * np.log(1 - y_predicted + epsilon))
        return cost
    def get_params(self):
        """Returns the learned weights and bias."""
        return self.weights, self.bias