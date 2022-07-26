# Cell 1
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Cell 2
iris = datasets.load_iris()
X, y = np.array(iris.data), np.array(iris.target)

# Cell 3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Cell 4

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, input_dim=4, activation = 'relu'),] + 
    [ tf.keras.layers.Dense(3, activation = 'relu') for _ in range(8)] +
    [tf.keras.layers.Dense(3, activation='softmax'),])

# Cell 5
net.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

# Cell 6
net.fit(X_train, y_train, epochs=150, batch_size=24)

# Cell 7
score = net.evaluate(X_test, y_test, verbose=0)
print("Test loss:", format(score[0],".4f"))
print("Test accuracy:", score[1])

# Cell 8
y_softmax = net.predict(X_test)

y_pc = np.argmax(y_softmax, axis = -1)

y_pred = keras.utils.to_categorical(y_pc)

Cmat = tf.math.confusion_matrix(y_test.argmax(axis=-1),y_pred.argmax(axis=-1))
print(Cmat)

