# Cell 5
import numpy as np
def softmax(X: np.ndarray) -> np.ndarray:
    exp = np.exp(X)
    return  (exp / np.sum(exp))

Z = np.around(softmax(np.array([3, -2, 1])), 3)
print(f'Z = {Z}')

# Cell 7
for index1, value1 in enumerate(Z, 1):
    for index2, value2 in enumerate(Z, 1):
        if index1 == index2:
            print(f'Z{index1} * (1 - Z{index2}) = {np.around(value1 * (1 - value2), 3)}')
        else:
            print(f'-Z{index1} * Z{index2} = {np.around(-value1 * value2, 3)}')

# Cell 9
def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sum(-y_true * np.log2(y_pred))

loss = cross_entropy(np.array([0, 0, 1]), np.array([0.876, 0.006, 0.118]))
loss = np.around(loss, 3)
print(f'loss = {loss}')

# Cell 10
# Setup for questions 3 and 4

from sklearn.model_selection import train_test_split, KFold

filename = 'winequality-white.csv'
data = np.genfromtxt(filename, delimiter=',')[1:]
X = (data[1:, :-1]).astype(float)
y = (data[1:, -1]).astype(int)

print(f'X.shape = {X.shape}')
print(f'y.shape = {y.shape}')

# Cell 11
# Setup continued, Train-Test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f'X_train.shape = {X_train.shape}')
print(f'y_train.shape = {y_train.shape}')
print(f'X_test.shape = {X_test.shape}')
print(f'y_test.shape = {y_test.shape}')

# Cell 12
# Setup continued, One hot encoding

import tensorflow as tf
from tensorflow import keras

y_train = keras.utils.to_categorical(y_train-3)
y_test = keras.utils.to_categorical(y_test-3)

print(f'y_train.shape = {y_train.shape}')
print(f'y_test.shape = {y_test.shape}')

# Cell 13
# Setup continued, K-Fold Validation splits
# Will use the same splits for Question 3 and 4

splits = KFold(n_splits=4).split(X_train, y_train)
splits = tuple(splits)

# Cell 14
# Question 3

for index, (training, validating) in enumerate(splits, 1):
    print(f'split {index}\n')
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8, input_dim=11, activation = 'relu'),
        tf.keras.layers.Dense(7, activation='softmax')])
    
    net.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    net.fit(X_train, y_train, epochs=150, batch_size=256, verbose=0)
    score = net.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss:{score[0]:.3f}')
    print(f'Test accuracy: {score[1] * 100:.3f}%')
    print('-' * 100)

# Cell 15
# Question 4

for index, (training, validating) in enumerate(splits, 1):
    print(f'split {index}\n')
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_dim=11, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'relu'),
        tf.keras.layers.Dense(7, activation='softmax')])
    
    net.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    net.fit(X_train, y_train, epochs=150, batch_size=256, verbose=0)
    score = net.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss:{score[0]:.3f}')
    print(f'Test accuracy: {score[1] * 100:.3f}%')
    print('-' * 100)

