# Cell 1
# Setup

import tensorflow.keras as keras
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import SGD

# Cell 2
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f'X_train.shape = {X_train.shape}')
print(f'y_train.shape = {y_train.shape}')
print(f'X_test.shape = {X_test.shape}')
print(f'y_test.shape = {y_test.shape}')

# Cell 3
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f'y_train.shape = {y_train.shape}')
print(f'y_test.shape = {y_test.shape}')

# Cell 4
net = Sequential() 
# the first convolutional layer 
net.add(Conv2D(8, (3, 3), padding='same', input_shape=(32,32, 3))) 
net.add(BatchNormalization()) 
net.add(Activation('relu')) 
 
net.add(MaxPooling2D((2, 2),strides=2)) 
 
# the second convolutional layer 
net.add(Conv2D(16, (3, 3), padding='same')) 
net.add(BatchNormalization()) 
net.add(Activation('relu')) 
 
net.add(MaxPooling2D((2, 2),strides=2)) 
 
# the third convolutional layer 
net.add(Conv2D(32, (3, 3), padding='same')) 
net.add(BatchNormalization()) 
net.add(Activation('relu')) 
 
# Classification 
net.add(Flatten()) 
net.add(Dense(64, activation='relu')) 
net.add(Dense(10, activation='softmax')) 
 
net.summary()

# Cell 5
net.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# Cell 6
epochs = 150
batch_size = 256
validation_split = 0.15

history = net.fit(
    x=X_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split
)

# Cell 7
import seaborn as sn
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_classified, y_true):
    # Compute confusion matrix
    c_mat = confusion_matrix(y_true, y_classified)

    group_counts = (f"{value:.0f}" for value in c_mat.flatten())
    group_percentages = (f"{value:.0f}" for value in c_mat.flatten() / np.sum(c_mat))
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(c_mat.shape[0], c_mat.shape[1])

    plt.figure(figsize=(12,10))
    sn.heatmap(c_mat, annot=labels, fmt='', cmap='rocket_r')
    plt.title("Confusion Matrix")
    plt.ylabel('Output Class')
    plt.xlabel('Target Class')

# Cell 8
# Evaluate the trained model using keras built-in function
score = net.evaluate(X_test, y_test)
print(f'Test loss: {score[0]:.3f}')
print(f'Test accuracy: {score[1]:.3f}') 

y_classified = np.argmax(net.predict(X_test), axis=1)
y_true =  np.argmax(y_test, axis=1)
# plot confusion matrix
plot_confusion_matrix(y_classified, y_true)

# Cell 9
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)

plt.plot(epochs, val_acc,'b')
plt.plot(epochs, acc,'r.')
plt.title("Training Accuracy")

plt.figure()
plt.plot(epochs, val_loss,'b')
plt.plot(epochs,loss,'r.')
plt.title("Training Loss")
plt.show()

