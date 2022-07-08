# Cell 5
# Question 2

# Setup
import os
os.environ["PATH"] += os.pathsep + "D:/Graphviz/bin/"

# %matplotlib inline
from matplotlib import pyplot as plt
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

filename = 'winequality-red.csv'
data = np.genfromtxt(filename, delimiter=',', dtype=str)
X, y = data[1:, :-1], data[1:, -1]
X = X.astype(float)
y = y.astype(float)

print(f'{X.shape = }')
print(f'{y.shape = }')

# Cell 6
# Question 2 continued

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

print(f'{X_train.shape = }')
print(f'{y_train.shape = }')
print(f'{X_test.shape = }')
print(f'{y_test.shape = }')

# Cell 7
# Question 2 continued

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'{y_pred.shape = }')

# Cell 8
# Question 2 continued

cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix = \n{cm}')
print(f'Apprarently, Labels at index 2 and 3 are most likely to be confused!')

# Cell 9
# Question 2 continued

dot_data = export_graphviz(
    model, 
    out_file=None,
    feature_names=data[0, :-1],
    class_names=data[:, -1].astype(str),
    filled=True,
    rounded=True,  
    special_characters=True
)
graph = graphviz.Source(dot_data) 
# Graph's too wide!!
graph

