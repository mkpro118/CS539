# Cell 1
# Setup

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.metrics import confusion_matrix
import numpy as np

filename = 'iris.csv'

data = np.genfromtxt(filename, delimiter=',')
X, y = data[:, :-1], data[:, -1]

print(f'{X.shape = }')
print(f'{y.shape = }')

# Cell 2
# Question 1 part (a)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
print(f'{X_train.shape = }')
print(f'{y_train.shape = }')
print(f'{X_test.shape = }')
print(f'{y_test.shape = }')

# Cell 3
# Question 1 part (b)

def models():
    yield from [kNN(n_neighbors=i).fit(X_train, y_train) for i in range(1, 10)]

# Cell 4
# Question 1 part (c)

classification_rates = []
    
for model in models():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = np.trace(cm) / np.sum(cm)
    classification_rates.append(acc)


best_k = np.argmax(classification_rates) + 1
best_acc = classification_rates[best_k - 1] * 100
print(f'Best classification rate was achieved with k = {best_k}: {best_acc:.5}%')


print('For other values of k')
for idx, val in enumerate(classification_rates, 1):
    if idx == best_k:
        continue
    print(f'Classification rate achieved with k = {idx}: {(val*100):.5}%')

