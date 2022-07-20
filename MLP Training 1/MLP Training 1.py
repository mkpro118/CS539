# Cell 1
# Question 1

import numpy as np

rng = np.random.default_rng()

filename = 'iris.csv'
data = np.genfromtxt(filename, delimiter=',')

X = data[:, :-1]
y = data[:, -1]
print(f'{X.shape = }')
print(f'{y.shape = }')

# Cell 2
def one_hot_encode(y:np.ndarray, n_classes: int, true_label: int) -> np.ndarray:
    one_hot = np.zeros((y.shape[0], n_classes))
    one_hot[y == true_label, 0] = 1
    one_hot[y != true_label, 1] = 1
    return one_hot

y = one_hot_encode(y, 2, 1)
print(f'{y.shape = }')

# Cell 3
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f'{X_train.shape = }')
print(f'{y_train.shape = }')
print(f'{X_test.shape = }')
print(f'{y_test.shape = }')

# Cell 4
def batches(X, y, batch_size=24):
    indices = np.arange(len(X))
    rng.shuffle(indices)
    for i in range(0, len(X), batch_size):
        idxs = indices[i : min(i + batch_size, len(X))]
        yield X[idxs], y[idxs]

# Cell 5
def softmax(X: np.ndarray) -> np.ndarray:
    exp = np.exp(X)
    return  (exp.T / np.sum(exp, axis=1)).T

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(-y_true * np.log(y_pred))

# Cell 6
weights = rng.normal(0, 0.01, (4, 2))
bias = rng.normal(0, 0.01, (2,))

print(f'{weights.shape = }')
print(f'Weights=\n{weights}')
print(f'\n{bias.shape = }')
print(f'Bias=\n{bias}')

# Cell 7
def predict(X: np.ndarray) -> np.ndarray:
    return softmax(X @ weights + bias)

def run_batch(X, y, lr):
    global weights, bias
    output = predict(X)
    e = output - y
    weights -= (lr / len(X)) * (X.T @ e)
    bias -= lr

def run_epoch(X, y, lr):
    for batch in batches(X, y):
        run_batch(*batch, lr)

def train(epochs: int = 5, learning_rate: float = 0.01) -> None:
    for epoch in range(1, epochs + 1):
        run_epoch(X_train, y_train, learning_rate)
        error = cross_entropy(y, predict(X))
        print(f'{epoch = :2d} | {error = }')

train(10, 0.02)

# Cell 8
print('Weights after training')
print(f'Weights=\n{weights}')
print(f'\nBias=\n{bias}')

# Cell 9
def score(y_true, y_pred):
    return 1 - np.mean(
        np.abs(
            np.argmax(y_true, axis=1) - np.argmax(y_pred, axis=1)
        )
    )

s = score(y_test, predict(X_test))
print(f'Score = {s*100:.2f}%')

