# Cell 1
# Question 1

# setup
import numpy as np

# Feature matrix and Label vectors
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

# Label filters
_y1 = y == 1
_y2 = y == 2

print(f'X = \n{X}')
print(f'y = {y}')

# Cell 2
# Question 1 part (a)

mean = np.mean(X, axis=0)
mean1 = np.mean(X[_y1], axis=0)
mean2 = np.mean(X[_y2], axis=0)

print(f'  Overall mean vector   = {mean}')
print(f'Mean vector for label 1 = {mean1}')
print(f'Mean vector for label 2 = {mean2}')

# Cell 3
# Question 1 part (b)

S1 = (x := (X[_y1] - mean1)).T @ x
S2 = (x := (X[_y2] - mean2)).T @ x

print(f'S1 =\n{S1}')
print(f'S2 =\n{S2}')

# Cell 4
# Question 1 part (c)

# Within cluster scattering matrix
Sw = S1 + S2

# Between cluster scattering matrix
m1 = (mean1 - mean).reshape((2, 1))
m2 = (mean2 - mean).reshape((2, 1))
Sb = np.sum(_y1) * (m1 @ m1.T) + np.sum(_y2) * m2 @ m2.T

print(f'Sw = \n{Sw}')
print(f'Sb = \n{Sb}')

# Cell 5
# Question 1 part (d)

Sw_inv_Sb = np.linalg.inv(Sw) @ Sb

# Eigen Value and Vectors
eig_val, eig_vec = np.linalg.eig(Sw_inv_Sb)

w_LDA = eig_vec[:, eig_val.argmax()].reshape((2, 1))
print(f'w_LDA =\n{w_LDA}')

# Cell 6
# Question 1 part (e)

y1 = w_LDA.T @ X[0].reshape((2, 1))
y2 = w_LDA.T @ X[1].reshape((2, 1))
y3 = w_LDA.T @ X[2].reshape((2, 1))
y4 = w_LDA.T @ X[3].reshape((2, 1))
y5 = w_LDA.T @ X[4].reshape((2, 1))
y6 = w_LDA.T @ X[5].reshape((2, 1))

print(f'{y1 = }')
print(f'{y2 = }')
print(f'{y3 = }')
print(f'{y4 = }')
print(f'{y5 = }')
print(f'{y6 = }')

# Cell 7
# Answer is not equivalent to the sklearn answer!!

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
print(LDA(n_components=1).fit_transform(X, y))

