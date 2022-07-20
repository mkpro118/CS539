# Cell 5
# Question 2

# Setup
import numpy as np

filename = 'Ex_FFnetData.csv'
data = np.genfromtxt(filename, delimiter=',')

weights1 = np.array([
    [-5, 2, -5],
    [-5, 5, -1],
])

bias1 = np.array([1, 5, 4])

z1 = data @ weights1 + bias1
print(f'{z1.shape = }')

# Cell 6
weights2 = np.array([
    [1, 1, 1]
]).T

bias2 = -2.5

z = z1 @ weights2 + bias2
print(f'{z.shape = }')

# Cell 7
print(z)

