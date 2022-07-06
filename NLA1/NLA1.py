# Cell 1
# Setup
import numpy as np

X = np.zeros((2,3,4)) # Used in Question 4 from Section 2.3.13

np.random.seed(0) # For reproducibility of results
_randint = np.random.randint # Used in Question 5 from Section 2.3.13

A = np.arange(20).reshape((5,4)) # Used in Question 6 from Section 2.3.13

# Cell 2
# Section 2.3.13 Question 4

# The answer to the question is 2, which is verified below
print(f'{len(X) = }')

# Cell 3
# Section 2.3.13 Question 5

# Yes, for a tensor X of arbitrary shape, the value of len(X)
# is always the length of the 0th axis of the tensor.
# For example: if we have a tensor T of shape (a, b, c, ... , n)
# the value of len(T) will be equal to `a`.

random_dimensions = tuple([_randint(1, 10) for i in range(_randint(4, 10))])
# random_dimensions = (6, 1, 4, 4, 8, 4, 6, 3) if numpy.random.seed(0)
# Warning: The above result has been computed on numpy==1.22.4
#          might not be the same on different versions of numpy

T = np.zeros(random_dimensions)

print(f'{T.shape = }', f'{T.shape[0] == len(T) = }', sep=' | ')

# Cell 4
# Section 2.3.13 Question 6

# The operation produces a ValueError
# The operation A.sum(axis=1) produces a vector with the same length
# as the dimension of the 0th axis of the matrix, and loses the matrix A's
# second dimension, effectively making it 0.
# The value error is raised because the arrays now cannot be broadcast together
# because, according to the numpy docs, broadcasting of arrays is only possible
# if they have equal dimensions, or if one dimension has length 1,
# starting from the rightmost dimension.
# Our dimensions are (5,4) and (5,), so comparing dimensions from the righmost dimension
# 4 [from (5,4)] and 5 [from (5,)] are neither equal, nor equal to 1, 
# so broadcast is not possible which causes numpy to raise a ValueError.

# Reference to Array Broadcasting: 
#     https://numpy.org/doc/stable/user/basics.broadcasting.html

# Note:
# As a counter example, `A.sum(axis=0)` will produce a vector of dimension (4,)
# Now comparing dimensions from the rightmost dimension,
# 4 [from (5,4)] and 4 [from (4,)] are equal, so they are brodcastable.
# and A / A.sum(axis=0) will work without any [runtime] error.

print(A)
print(A.sum(axis=0))

A_sum_axis_1 = A / A.sum(axis=1)

# Cell 5
# Question:
# When traveling between two points in Manhattan, New York City, what is the distance that 
# you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you 
# travel diagonally? This kind of distance is what type of norm we learned in this module?

# Answer:
# The distance covered is equal to the sum of the length of the avenues and streets 
# that we have to travel through to reach from one point to another. 
# Mathematically, distance (d) in terms of avenues (a) and streets (s) is,
# d = sum(|a|) + sum(abs|s|) 
# Realistically, its not possible to travel diagonally between two points in Manhattan, due to the
# presence of path obstructive infrastructure. We are limited to travelling between two points using
# the streets and avenues in the city.
# This is the L1 norm that we learned in this module, where the norm is the sum of the absolute values.

