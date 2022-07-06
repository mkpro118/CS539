# Cell 1
# Setup

import numpy as np

A = np.array([1, -2, 3, 2, 1, -1,]).reshape((2,3)).T
A

# Cell 2
# Question 1 part (a)
# Expected answer is 2, since matrix A has 2 linearly independent column vectors

rank = np.linalg.matrix_rank(A)
print(f'{rank = }') 

# Cell 3
# Question 1 part (b)
# The dimensions of U (left singular vectors) are (3 x 2)
# The dimensions of S (singular value matrix) are (2 x 2)
# The dimensions of V (right singular vectors) are (2 x 2)
# K = min(A.shape) => K = min(3, 2) = 2

# Computing the SVD
# The full_matrices parameter has been set to False to only compute
# the relevant/important vectors in the U,S and V matrices
# (not the vectors that will result to 0 after computation)
U, S, V = np.linalg.svd(A, full_matrices=False)

# Make S a diagonal matrix
S = np.diag(S)

# Verification of the dimensions of U (left singular vectors)
print(f'Dimensions of U = {U.shape}')

# Verification of the dimensions of S (singular value matrix)
print(f'Dimensions of S = {S.shape}')

# Verification of the dimensions of V (right singular vectors)
print(f'Dimensions of V = {V.shape}')

# Cell 4
# Question 1 part (b) continued
u1 = U[:,0].reshape((U.shape[0], 1))
u2 = U[:, 1].reshape((U.shape[0], 1))

print(f'U =\n{U}', end='\n\n')
print(f'{u1 = }', end='\n\n')
print(f'{u2 = }')

# Cell 5
# Question 1 part (b) continued
s1, s2 = S[S != 0]

print(f'S =\n{S}', end='\n\n')
print(f'{s1 = }', end='\n\n')
print(f'{s2 = }')

# Cell 6
# Question 1 part (b) continued
v1 = V[0, :].reshape((1, V.shape[1]))
v2 = V[1, :].reshape((1, V.shape[1]))

print(f'V =\n{V}', end='\n\n')
print(f'{v1 = }', end='\n\n')
print(f'{v2 = }')

# Cell 7
# Question 1 part (b) continued
# Verification that u1, u2, s1, s2, v1 and v2 are correct

# Summation to compute A from the SVD [@ is equivalent to np.dot()]
A_from_svd = (s1 * (u1 @ v1)) + (s2 * (u2 @ v2))

print(f'A from SVD =\n{A_from_svd}', end='\n\n')

# Some values here are false due to inexact floating point number comparison
# However, the array printed above should be representative of the correctness
# of the computation.
A_from_svd_compared = A_from_svd == A
print(f'A from SVD compared to original A =\n{A_from_svd_compared}', end='\n\n')

# We can use the numpy.isclose() method to verify that the values are indeed almost equal
are_they_close = np.isclose(A, A_from_svd)
print(f'A from SVD compared to original A using numpy.isclose() =\n{are_they_close}')

# Cell 8
# Question 1 part (c)

# Computing B
B = A @ A.T
print(f'B =\n{B}')

# Cell 9
# Question 1 part (c) continued
eigen_values, eigen_vectors = np.linalg.eig(B)
eigen_vectors_T = eigen_vectors.T
print(f'Eigen Values =\n{eigen_values}')
print(f'Eigen Vectors =\n{eigen_vectors}')
print(f'Eigen Vectors Transpose =\n{eigen_vectors_T}')

# Cell 10
# Question 1 part (c) continued

# W, W_T, L correspond to W, W.T and Lambda in the question
# M = 3, corresponding to the number of rows or columns in the square matrix B

W = eigen_vectors
W_T = eigen_vectors_T
L = np.diag(eigen_values)

M = B.shape[0] # number of rows
print(f'{M = }')

# Cell 11
# Question 1 part (c) continued
w1 = W[:, 0].reshape((W.shape[0], 1))
w2 = W[:, 1].reshape((W.shape[0], 1))
w3 = W[:, 2].reshape((W.shape[0], 1))

print(f'w1 =\n{w1}', end='\n\n')
print(f'w2 =\n{w2}', end='\n\n')
print(f'w3 =\n{w3}')

# Cell 12
# Question 1 part (c) continued
l1, l2, l3 = L[L != 0]

print(f'{l1 = }')
print(f'{l2 = }')
print(f'{l3 = }')

# Cell 13
# Question 1 part (c) continued
wt1 = W_T[0, :].reshape((1, W_T.shape[0]))
wt2 = W_T[1, :].reshape((1, W_T.shape[0]))
wt3 = W_T[2, :].reshape((1, W_T.shape[0]))

print(f'w.T1 =\n{wt1}', end='\n\n')
print(f'w.T2 =\n{wt2}', end='\n\n')
print(f'w.T3 =\n{wt3}')

# Cell 14
# Question 1 part (c) continued
# Verification that w1, w2, w3, l1, l2, l3, wt1, wt2 and wt3 are correct

# Summation to computer from B the Eigen Decomposition
B_from_eigen_decompostion = (l1 * (w1 @ wt1)) + (l2 * (w2 @ wt2)) + (l3 * (w3 @ wt3))
print(f'B from Eigen Decompostion =\n{B_from_eigen_decompostion}', end='\n\n')

# Similar to the SVD, values here are false due to precision errors of floating point numbers
B_from_eigen_decompostion_compared = B_from_eigen_decompostion == B
print(f'B from Eigen Decomposition compared to original B =',
      f'{B_from_eigen_decompostion_compared}', sep='\n', end='\n\n')

# We can use the numpy.isclose() method to verify that the values are indeed almost equal
are_they_close2 = np.isclose(B, B_from_eigen_decompostion)
print(f'B from Eigen Decomposition compared to original B using numpy.isclose() =',
      f'{are_they_close2}', sep='\n')

# Cell 16
# Question 2 part (b)
# r's sign represents if it's on the same or opposite side of the origin
w = np.array([4, 5]) / (41**0.5)
b = -20 / (41 ** 0.5)

def g(x: np.ndarray) -> float:
    return w[0]*x[0] + w[1]*x[1] + b

def r(x: np.ndarray) -> float:
    # np.hypot computes the hypotenuse of a right triangle, equivalent to Eucleadean distance
    return g(x) / np.hypot(*w) 
print(r(np.array([4.5, 3])))

# Cell 17


