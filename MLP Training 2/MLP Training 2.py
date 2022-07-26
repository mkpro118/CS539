# Cell 1
# Question 1

# Setup
import numpy as np

X = np.array([[1,1,1,], [1, 2, 3,], [2, -1, 0,],]).T
y = np.array([[0, 1, 0,],]).T
w1 = np.array([[1, 2, 0,], [-1, 0, 3,],]).T / 10
w2 = np.array([[5e-2, -1e-1, 2e-1,],]).T

print(f'{X.shape = }')
print(f'{y.shape = }')
print(f'{w1.shape = }')
print(f'{w2.shape = }')

# Cell 2
def square_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def sigmoid(X: np.ndarray) -> np.ndarray:
    return (np.exp(-X) + 1) ** -1

# Cell 3
U1 = np.around(X @ w1, 3)
print(f'U1\n{U1}')

# Cell 4
Z1 = np.around(sigmoid(U1), 3)
print(f'Z1\n{Z1}')
Z1 = np.hstack(
    (
        np.array([[1,1,1,],]).T,
        Z1
    )
)

# Cell 5
U2 = np.around(Z1 @ w2, 3)
print(f'U2:\n{U2}')

# Cell 6
Z2 = np.around(sigmoid(U2), 3)
print(f'Z2:\n{Z2}')

# Cell 7
delta_l2 = np.around(Z2 * (1-Z2) * (Z2 - y), 3)
print(f'Delta layer 2\n{delta_l2}')

# Cell 8
delta_l1 = np.around(delta_l2 @ w2[1:].T, 3)
print(f'Delta layer 1\n{delta_l1}')

# Cell 9
dw2 = np.around(Z1.T @ delta_l2, 3)
print(f'Layer 2 weight gradient\n{dw2}')

# Cell 10
dw1 = np.around(X.T @ delta_l1, 3)
print(f'Layer 1 weight gradient\n{dw1}')

