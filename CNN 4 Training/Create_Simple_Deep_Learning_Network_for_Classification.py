# Cell 2
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# Load mnist images dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# categorical label, convert to one_hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Cell 3
# plot first few images
for i in range(20):
	plt.subplot(4,5,i+1)
	plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
plt.show()

# Cell 5
from sklearn.model_selection import train_test_split

# Divide the data into 75% training and 25% validation data sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, shuffle=True, random_state=0)

# summarize loaded dataset
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Val  : X=%s, y=%s' % (X_val.shape, y_val.shape))
print('Test : X=%s, y=%s' % (X_test.shape, y_test.shape))

# Cell 7
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Activation,Dense,Flatten

# Cell 8
# NOTE: you can play around with normalization before or after ReLU activation
net = Sequential()
net.add(Conv2D(8, (3, 3), padding='same', input_shape=(28, 28, 1)))
# net.add(BatchNormalization())
net.add(Activation('relu'))

net.add(MaxPooling2D((2, 2),strides=2))

net.add(Conv2D(16, (3, 3), padding='same'))
# net.add(BatchNormalization())
net.add(Activation('relu'))

net.add(MaxPooling2D((2, 2),strides=2))

net.add(Conv2D(32, (3, 3), padding='same'))
# net.add(BatchNormalization())
net.add(Activation('relu'))

net.add(Flatten())
net.add(Dense(10, activation='softmax'))

# Cell 10
from tensorflow.keras.optimizers import SGD

# Cell 11
# Hyperparameters
lr = 0.01
mom = 0.9 
ep = 10
bs = 100

# Cell 12
# compile and fit the keras model
opt = SGD(learning_rate=lr, momentum=mom)
net.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = net.fit(X_train, y_train, epochs=ep, batch_size=bs, validation_data=(X_val,y_val), verbose=1)

# Cell 14
# You can visualize the results with a confusion matrix.
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
def plot_confusion_matrix(y_classified, y_true):
  # Compute confusion matrix
  c_mat = np.zeros((y_test.shape[1],y_test.shape[1]))
  for i in range(len(y_true)):
    c_mat[y_classified[i], y_true[i] ] += 1

  group_counts = ["{0:0.0f}".format(value) for value in c_mat.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in c_mat.flatten()/np.sum(c_mat)]
  labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
  labels = np.asarray(labels).reshape(c_mat.shape[0], c_mat.shape[1])

  plt.figure(figsize=(12,10))
  sn.heatmap(c_mat, annot=labels, fmt='', cmap='rocket_r')
  plt.title("Confusion Matrix")
  plt.ylabel('Output Class')
  plt.xlabel('Target Class')

# Cell 15
# Evaluate the trained model using keras built-in function
score = net.evaluate(X_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1]) 

y_classified = np.argmax(net.predict(X_test), axis=1)
y_true =  np.argmax(y_test, axis=1)
# plot confusion matrix
plot_confusion_matrix(y_classified, y_true)

