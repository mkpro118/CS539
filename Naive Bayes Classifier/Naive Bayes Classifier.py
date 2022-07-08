# Cell 1
# Question 1

# Setup
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

filename = 'winequality-red.csv'

data = np.genfromtxt(filename, delimiter=',')
X = data[1:, :-1]
y = data[1:, -1]

print(f'{X.shape = }')
print(f'{y.shape = }')

# Cell 2
# Question 1 continued
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f'{X_train.shape = }')
print(f'{y_train.shape = }')
print(f'{X_test.shape = }')
print(f'{y_test.shape = }')

# Cell 3
# Question 1 continued
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'{y_pred.shape = }')

# Cell 4
# Question 1 continued
acc = accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy Score={acc:.5}%')
print(f'Confusion Matrix =\n{cm}')

