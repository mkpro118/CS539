# Cell 1
# Question 1

# Setup
import numpy as np
from sklearn.metrics import r2_score

# Cell 2
# To apply linear regression

class LinearRegression:
    @staticmethod
    def apply(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''
        Apply Linear Regression
        '''
        return (X @ w) + b

# Cell 3
# To compute loss using MSE

class MeanSquaredError:
    @staticmethod
    def apply(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        '''
        Apply MSE loss
        '''
        return ((y_pred - y_true) ** 2) / len(y_true)

# Cell 4
# Single Neuron Model

class SNM:
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SNM':
        '''
        Fit the model with training features and labels
        Compute initial (random) weights and biases
        '''
        self.X = X
        self.y = y
        self.sample_count = len(X)
        if len(y) != self.sample_count:
            raise ValueError(
                "Number of Samples is not equal to number of labels"
            )
        if self.y.ndim == 1:
            self.y = self.y.reshape((-1, 1))
        
        self.rng = np.random.default_rng()
        self.weights = self.rng.normal(0, 1e-2, (X.shape[-1])).reshape((-1, 1))
        self.bias = self.rng.normal(0, 0.01, 1)
        return self
    
    def train(self, batch_size: int = 24, epochs: int = 3,
              learning_rate:float = 1e-2, verbose: bool = False) -> None:
        '''
        Train the model with stochastic gradient descent
        for given number of epochs and batch size
        '''
        self.learning_rate = learning_rate
        self.batch_size = min(batch_size, self.sample_count)
        for epoch in range(epochs):
            self._run_epoch()
            loss = MeanSquaredError.apply(self.y, self.predict(self.X))
            loss = np.sum(loss)
            if verbose:
                print(f'Epoch {epoch + 1}, {loss = }')
                
    def _run_epoch(self):
        '''
        Run all iterations for one epoch
        '''
        for X, y in self._batches():
            self._run_batch(X, y)
    
    def _run_batch(self, X, y):
        '''
        Run an iteration using a minibatch
        Update the weights and biases with SGD
        '''
        output = self.predict(X)
        # e of the e * p hadamard product [e = z(k) - y(k)]
        e = output - y
        # p = 1 for linear activation
        p = 1
        self.weights -= (self.learning_rate / len(X)) * (X.T @ (e * p))
        self.bias -= self.learning_rate
    
    def predict(self, X):
        '''
        Predict the output of this SNM
        '''
        return LinearRegression.apply(X, self.weights, self.bias)
    
    def _batches(self):
        '''
        Get random mini batches of training data
        '''
        indices = np.arange(self.sample_count)
        self.rng.shuffle(indices)
        for i in range(0, self.sample_count, self.batch_size):
            idxs = indices[i : min(i + self.batch_size, self.sample_count)]
            yield self.X[idxs], self.y[idxs]

# Cell 5
filename = 'iris.csv'

data = np.genfromtxt(filename, delimiter=',')
X = data[:, :3]
y = data[:, 3]

snm = SNM()
snm.fit(X, y)
snm.train(verbose=True)

print(f'Weights are = {snm.weights.flatten()}')
print(f'Bias is = {snm.bias.flatten()}')

# Cell 6
r2 = r2_score(y.flatten(), snm.predict(X).flatten())
r2 = r2 * 100

print(f'R2 Score = {r2:.3f}%')

