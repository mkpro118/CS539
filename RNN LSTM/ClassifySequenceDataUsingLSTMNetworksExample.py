# Cell 3
from numpy.lib.shape_base import column_stack
import numpy as np
# Load the file
# f = open('ae.train','r')
with open('ae.train', 'r') as f: 
  contents = f.read()
  contents = contents.split("\n\n")

# number of time series 
N = len(contents) - 1

Xtrain = [[]] * (N)
for i in range(N):
  cols = contents[i].split('\n')
  Xtrain[i] = []
  for j in range(len(cols)):
    rows = cols[j].split(' ')[0:-1]
    for k in range(len(rows)):
      rows[k] = float(rows[k])
    Xtrain[i].append(rows)
  Xtrain[i] = np.array(Xtrain[i]).T

# Labels should be in blocks of 30
Ytrain = np.zeros((N,9))
for i in range(9):
  for j in range(30):
    Ytrain[30*i + j,i] = 1

# Print the shapes of the first 5 time series, 270 time series total
for i in range(5):
  print("Sample "  + str(i) + " shape: "+ str(np.shape(Xtrain[i]))) 

# Cell 5
import matplotlib.pyplot as plt

plt.plot(Xtrain[0].T)
plt.xlabel('Time Step')
plt.title('Training Observation 1')
numFeatures = np.shape(Xtrain[0])[0]
leg_string = []
for i in range(numFeatures):
  leg_string.append("Feature " + str(i))
plt.legend(leg_string, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Cell 7
numObservations = len(Xtrain)
seq_len = []
for i in range(numObservations):
  sequence = Xtrain[i]
  seq_len.append(np.shape(Xtrain[i])[1])

max_seq_len = np.max(seq_len)

# Cell 8
# Create data generator to pad and format the data.
dimension = 12
special_value = -10
Xpad = np.full((N, max_seq_len, dimension), fill_value=special_value).astype(float)
for s, x in enumerate(Xtrain):
  seq_len = x.shape[1]
  Xpad[s, 0:seq_len, :] = x.T

Ytrain = np.array(Ytrain)
print("X shape: ")
print(np.shape(Xpad))
print("Y shape: ")
print(np.shape(Ytrain))

# Cell 10
import keras
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

dimension = 12
numHiddenUnits = 100
numClasses = 9


# Specify our layer structure
model = Sequential()
#model.add(Embedding(input_size, input_size, input_length=))
model.add(Masking(mask_value=special_value, input_shape=(None, dimension)))
#model.add(keras.Input(shape=(dimension,1)))
model.add(Bidirectional(LSTM(numHiddenUnits)))
model.add(Dense(numClasses, activation='sigmoid'))
model.add(Dense(numClasses, activation='softmax'))


# Cell 12
maxEpochs = 100
miniBatchSize = 27

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Cell 14
model.fit(Xpad, Ytrain, epochs=maxEpochs, batch_size=miniBatchSize)

# Cell 16
# Load the file
# f = open('ae.test','r')

with open('ae.test', 'r') as f: 
  contents = f.read()
  contents = contents.split("\n\n")

Ntest = len(contents) - 1
Xtest = [[]] * (Ntest)
for i in range(Ntest):
  rows = contents[i].split('\n')
  Xtest[i] = []
  for j in range(len(rows)):
    cols = rows[j].split(' ')[0:-1]
    for k in range(len(cols)):
      cols[k] = float(cols[k])

    Xtest[i].append(cols)
  

  Xtest[i] = np.array(Xtest[i]).T


# f = open('size_ae.test')
with open('size_ae.test', 'r') as f: 
  contents = f.read()
  contents = contents.split(" ")[0:-1]
  contents = [int(i) for i in contents]

# Labels are specified by sizeae.test
Ytest = np.zeros((Ntest,9))
cc= 0
for i in range(9):
  for j in range(contents[i]):
    Ytest[cc,i] = 1
    cc = cc + 1

# Cell 17
numObservations = len(Xtest)
seq_len = []
for i in range(Ntest):
  sequence = Xtest[i]
  seq_len.append(np.shape(Xtest[i])[1])

#max_seq_len = 29
max_seq_len = np.max(seq_len)

# Cell 19
# Create data generator to pad and format the data.
dimension = 12
special_value = -10
XPad2 = np.full((Ntest, max_seq_len, dimension), fill_value=special_value).astype(float)
for s, x in enumerate(Xtest):
  seq_len = x.shape[1]
  XPad2[s, 0:seq_len, :] = x.T

Ytest = np.array(Ytest)
print(np.shape(XPad2))
print(np.shape(Ytest))

# Cell 21
output = model.predict(XPad2)
classified = np.argmax(output,axis=1)
yhat = np.zeros((Ntest,9))
for i in range(len(classified)):
  yhat[i,classified[i]] = 1

# Cell 23
acc = np.sum(np.multiply(yhat, Ytest))/Ntest
print("Accuracy: " + str(acc))

# Cell 25
loss, acc2 = model.evaluate(XPad2,Ytest)
print("Accuracy: " + str(acc2))

