# Cell 2
# Question 2

import numpy as np
from sklearn.model_selection import train_test_split

filename = 'winequality-white.csv'
data = np.genfromtxt(filename, dtype='f4', delimiter=',')

X = data[1:, :-1]
y = data[1:,  -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=None)

# Cell 3
mean_scaled = np.mean(X_train, axis=0)
std_scaled = np.std(X_train, axis=0)

def normalize(X: np.ndarray) -> np.ndarray:
    return (X - mean_scaled) / std_scaled

X_train = normalize(X_train)
X_test = normalize(X_test)

# Cell 4
# Gating Network initiation: Using kmeans algorithm to partition the data into 3 clusters
from sklearn.cluster import KMeans

C = 7
kmeans = KMeans(n_clusters=C, random_state=0).fit(X_train)
idx = kmeans.labels_

# Cell 5
# Traing the gating network model using a naive baysian classifier using the clustering indices obtained after clustering
from sklearn.naive_bayes import GaussianNB

gNet = GaussianNB().fit(X_train, idx)

# Cell 6
# Traing individual expert classifiers using each cluster of training data
from sklearn.tree import DecisionTreeClassifier

Exp = []
for k in range(C):
    Exp.append(DecisionTreeClassifier(random_state=0).fit(X_train[idx==k], y_train[idx==k]))

# Cell 7
# Validation with X_test, y_test
from keras.utils.np_utils import to_categorical   

# Gating network output
y_gNet = gNet.predict(X_test)
y_gNet_1h = to_categorical(y_gNet)


# Let each expert predict for the entire validation data set
y_exp = []
for k in range(C):
    y_exp.append(Exp[k].predict(X_test))

# Selecting the prediction from corresponding expert assigned by the gating network
y_exp_T = np.array(y_exp).T
y_gated = np.sum(y_exp_T * y_gNet_1h, axis=1)

# Cell 8
# Check results
from sklearn.metrics import confusion_matrix

# expert classifier's performance w.r.t. the entire validation dataset
for k in range(C):
    print(f'Expert #{k}\n',confusion_matrix(y_test,y_exp[k]))

print('Gated \n', confusion_matrix(y_test,y_gated))

# Cell 9
# find which expert made the wrong classification
print('expert #',y_gNet[y_gated != y_test],'made incorrect prediction')

