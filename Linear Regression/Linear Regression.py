# Cell 1
# Question 1

# Setup
# %matplotlib inline
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

plt.style.use('dark_background')

filename = 'iris.csv'
data = np.genfromtxt(filename, delimiter=',')

X = data[:, :3]
y = data[:, 3]

print(f'{X.shape = }')
print(f'{y.shape = }')

# Cell 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(f'{X_train.shape = }')
print(f'{y_train.shape = }')
print(f'{X_test.shape = }')
print(f'{y_test.shape = }')

# Cell 3
model = LinearRegression()

model.fit(X_train, y_train)

# Cell 4
# Question 1 part (a)

intercept = model.intercept_

slopes = np.around(model.coef_, 3)

print(f'{intercept = :.3f}')
print(f'{slopes = }')

# Cell 5
# Question 1 part (b)

R2_score = model.score(X_test, y_test)
print(f'{R2_score = }')

# Cell 6
# Question 1 part (c)

adjusted_X_train = np.sum(slopes * X_train, axis=1)
adjusted_X_test = np.sum(slopes * X_test, axis=1)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax.scatter(adjusted_X_train, y_train, c='royalblue')
ax.plot(adjusted_X_test, model.predict(X_test), c='hotpink', linewidth=3)

plt.show()

