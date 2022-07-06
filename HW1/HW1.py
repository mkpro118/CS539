# Cell 1
# Setup
import numpy as np

# Cell 2
# Question 1 part (a)

# Column vector a
a = np.array([1, -2, 3, 2]).reshape((4, 1))

# Column vector b
b = np.array([1, -2, 3, 2]).reshape((4, 1))

# Computing c = a @ b.T
c = a @ b.T
print(f'Dimensions of c = {c.shape}')
print(f'c =\n{c}')

# Computing d = a.T @ b
d = a.T @ b
print(f'Dimensions of d = {d.shape}')
print(f'd = {d}')

# Cell 3
# Question 1 part (b)

# If the equation E = ADB = sum(d[i, i] * (a[i] @ b[i].T)) is true,
# it should be true for any random value in the matrices.
# This is obviously not a proof, we could just get lucky,
# the proof is given after this verification script on random numbers

# Matrix A, a (3, 2) matrix
A = np.random.random((3, 2))
print(f'A =\n{A}', end='\n\n')

# Matrix D, a (2, 2) diagonal matrix
D = np.diag(np.random.random((2, )))
print(f'D =\n{D}', end='\n\n')

# Matrix B, a (2, 4) matrix
B = np.random.random((2, 4))
print(f'B =\n{B}', end='\n\n')

# E is a list of two matrices
# E[0] is the matrix computed using E = A @ D @ B
# E[1] is the matrix computed using E = sum(d[i, i] * (a[i] @ b[i].T))
E = [None] * 2

E[0] = A @ D @ B
print(f'E[0] = \n{E[0]}', end='\n\n')

# To compute E, well add elements according to the formula,
# starting from a Null matrix of dimensions (3, 4)
# A @ D @ B => [(3, 2) @ (2, 2)] @ (2, 4) => (3, 2) @ (2, 4)
# A @ D @ B => (3, 4)
E[1] = np.zeros((3, 4))

# Performing the summation
for i in range(2): # indexes start from 0 instead of 1
    # D[i, i] selects the diagonal elements
    D_i = D[i, i]
    
    # A[:, i] selects the ith columns from A 
    # reshaping to make it compatible with mat_mul
    A_i = (_ := A[:, i]).reshape((_.shape[0], 1))
    
    # B[i] selects the ith row from B
    # reshaping to make it compatible with mat_mul
    B_i = (_ := B[i]).reshape((1, _.shape[0], ))
    
    # Summing after the operations
    E[1] += D_i * (A_i @ B_i)

print(f'E[1] = \n{E[1]}', end='\n\n')

# We can check if the matrices E[0] and E[1] are equal
# To avoid errors arising from floating point numbers,
# we will use the numpy.isclose() method
equal_Es = np.isclose(*E)

print(f'E[0] == E[1] =\n{equal_Es}', end='\n\n')

if equal_Es.all():
    print('Numerically, this is accurate for randomly generated number')
else:
    print(
        'Numerically, this did not work for',
        f'A =\n{A}',
        f'D =\n{D}',
        f'B =\n{B}',
        sep='\n'
    )

# Cell 5
# Question 1 part (c)

a = np.arange(20)
A = a.reshape((5, 4))
print(f'A =\n{A}')

# Cell 6
# Question 1 part (d)

hadamard_product = A * A
print(f'Hadamard Product =\n{hadamard_product}')

# Cell 7
# Question 2 part (a)

b = np.arange(24)
B = b.reshape((2, 3, 4))

print(f'B = \n{B}')

# Cell 8
# Question 2 part (b)

B_sum = np.sum(B)

print(f'Sum of elements in B = {B_sum}')

# Cell 9
# Question 2 part (c)

C, D = B # obtaining C and D via tuple unpacking

print(f'C =\n{C}')
print(f'D =\n{D}')

# Cell 11
# Setup
import math

table = {
    "Outlook": [
        'Sunny', 'Sunny', 'Overcast',
        'Rain', 'Rain', 'Rain', 'Overcast',
        'Sunny', 'Sunny', 'Rain', 'Sunny',
        'Overcast', 'Overcast', 'Rain',
    ],

    "Temperature": [
        "Hot", "Hot", "Hot", "Mild", "Cool",
        "Cool", "Cool", "Mild", "Cool", "Mild",
        "Mild", "Mild", "Hot", "Mild",
    ],

    "Humidity": [
        "High", "High", "High", "High", "Normal",
        "Normal", "Normal", "High", "Normal",
        "Normal", "Normal", "High", "Normal", "High",
    ],

    "Wind": [
        "Weak", "Strong", "Weak", "Weak", "Weak",
        "Strong", "Strong", "Weak", "Weak", "Weak",
        "Strong", "Strong", "Weak", "Strong",
    ],

    "Played": [
        "No", "No", "Yes", "Yes", "Yes", "No", "Yes",
        "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No",
    ],
}

# Function that takes in iterables and predicates (conditions) to
# filter values by the predicate and return the required probability
def pr(iterable: 'Iterable[str]', predicate: callable) -> float:
    return sum((1 for i in iterable if predicate(i))) / len(iterable)

# Cell 12
# Question 3 part (a)

# Finding probabilty of High Humidity
p_high = pr(table["Humidity"], lambda x: x == "High")
print("Pr. {Humidity = High} = ", p_high)

# Cell 13
# Question 3 part (b)

# Finding probabilty of a Sunny Outlook and Normal Humidity
p_sunny_normal = pr(
    list(zip(table["Outlook"], table["Humidity"])),
    lambda x: x[0] == "Sunny" and x[1] == "Normal"
)
# Rounded to 6 decimal places
print("Pr. {Outlook = Sunny AND Humidity = Normal} = ", f'{p_sunny_normal:.6}')

# Cell 14
# Question 3 part (c)

# Finding probabilty of Cold Temperatures or Weak Winds
p_cold_weak = pr(
    list(zip(table["Temperature"], table["Wind"])),
    lambda x: x[0] == "Cold" or x[1] == "Weak"
)
# Rounded to 6 decimal places
print("Pr. {Temperature = Cold OR Wind = Weak} = ", f'{p_cold_weak:.6}')

# Cell 15
# Question 3 part (d)

# Finding probabilty of playing football given Humidity is high
# We can find this using the formula,
# P(A|B) = P(A and B) / P(B)

p_played_high = pr(
    list(zip(table["Played"], table["Humidity"])),
    lambda x: x[0] == "Yes" and x[1] == "High" # P(A and B)
) / pr(table["Humidity"], lambda x: x == "High") # P(B)

# Rounded to 6 decimal places
print("Pr. {Play = Yes | Humidity = High} = ", f'{p_played_high:.6}')

# Cell 16
# Question 3 part (e)

# Finding probabilty of high Humidity, given football was played
# We can find this using the formula,
# P(A|B) = P(A and B) / P(B)

p_high_played = pr(
    list(zip(table["Humidity"], table["Played"])),
    lambda x: x[0] == "High" and x[1] == "Yes" # P(A and B)
) / pr(table["Played"], lambda x: x == "Yes") # P(B)

# Rounded to 6 decimal places
print("Pr. {Play = Yes | Humidity = High} = ", f'{p_high_played:.6}')

# Cell 17
# Question 3 part (f)

# Finding the entropy of the outcome of play

def information(x: float) -> float:
    return -math.log(x)

def entropy(pdf: callable, outcomes: set[str]) -> float:
    return sum(((p := pdf(x)) * information(p) for x in outcomes))

entropy_play = entropy(
    lambda x: pr(table["Played"], lambda y: x == y),
    set(table["Played"])
)

# Rounded to 7 decimal places
print("Entropy of the outcome of play = ", f'{entropy_play:.7}')

# Cell 18
# Question 4 

# Setup
filename = 'iris.csv'
iris = np.genfromtxt(filename, delimiter=",")
means = dict()
covariances = dict()

# Cell 19
# Question 4 part(a)
# Calcuating the mean matrix

# Mean vector for each label
for x, y in enumerate(range(3), 1):
    mean = np.mean(iris[y*50:(y+1)*50, :-1], axis=0)
    print(
        *(f'Mean for Label {x}\'s Feature {i}: {m:.7}'
          for i, m in enumerate(mean, 1))
        , sep='\n', end='\n\n'
    )
    means[x] = mean

# Cell 20
# Question 4 part (a) continued
# Calculating the covariance matrix

def E(X: np.ndarray) -> float:
    '''
    Params:
        | X: np.ndarray | The vector to compute the expectation for
    Returns:
        float: expected value of X
    
    This function computes the expected value for
    the random variable X.
    Since X is a vector of discrete values,
    Expectation of X is equal to the mean of X
    '''
    return np.mean(X)

def covariance(A: np.ndarray, B: np.ndarray) -> float:
    '''
    Params:
        | A: np.ndarray | Vector of Random Variable A values
        | B: np.ndarray | Vector of Random Variable B values
    Returns:
        float: Covariance of A and B
    
    This function computes the covariance between A and B
    using the formula
    Cov(A,B) = E(AB) - E(A)E(B)
    Cov: Covariance
     E : Expectation
    '''
    return E(A*B) - E(A) * E(B)

def cov(X: np.ndarray) -> np.ndarray:
    '''
    Params:
        | X: np.ndarray | The 2D array to calculate covariance for
    Returns:
        np.ndarray: The covariance matrix
    
    Calculates the covariance matrix for the given matrix X
    '''
    # number of features, taken as the second dimension
    # f = len(axis=1) = X.shape[1]
    f = X.shape[1]
    m = np.zeros((f, f))
    for i in range(m.shape[0]):
        m[i, i] = covariance(X[:, i], X[:, i])
        for j in range(i):
            m[i,j] = covariance(X[:, i], X[:, j])
            m[j, i] = m[i, j]
    return m

for x, y in enumerate(range(3), 1):
    c = cov(iris[y * 50 : (y+1) * 50, :-1])
    print(f'Covariance Matrix for Label {x} =', c, sep='\n', end='\n\n')
    covariances[x] = c

# Cell 21
# About Q4(a), The covariance matrix is not equivalent to
# the one computed by numpy.cov(). Each value computed by
# the numpy's cov function is slightly higher than
# the ones computed by the method I have used.
# What could be the cause of the discrepancy?

# Matrices computed by me
print("Matrices computed by me")
for x, y in enumerate(range(3), 1):
    c1 = cov(iris[y * 50 : (y+1) * 50, :-1])
    print(c1, end='\n\n')

print('-'*30, end='\n\n')

# Matrices computed by numpy.cov()
print("Matrices computed by numpy.cov()")
for x, y in enumerate(range(3), 1):
    c2 = np.cov(iris[y * 50 : (y+1) * 50, :-1].T)
    print(c2, end='\n\n')

print('-'*30, end='\n\n')
    
# The comparison
print("The comparison, using numpy.isclose()")
for x, y in enumerate(range(3), 1):
    c1 = cov(iris[y * 50 : (y+1) * 50, :-1])
    c2 = np.cov(iris[y * 50 : (y+1) * 50, :-1].T)
    
    # All entries are false
    print(np.isclose(c1, c2), end='\n\n')

# Cell 22
# Question 4 part (b)

# Setup
xs = {
    1: iris[0, :-1],
    2: iris[50, :-1],
    3: iris[100, :-1],
}

# Function to compute multivariate normal
def mvn(x: np.ndarray, m: np.ndarray, c: np.ndarray) -> float:
    '''
    Params:
        | x: np.ndarray | Feature vector
        | m: np.ndarray | Mean vector
        | c: np.ndarray | Covariance Matrix
    Returns:
        float: Multivariate Normal
    
    This function computes and returns the multivariate normal
    of the feature vector x, given it's mean vector 
    and covariance matrix according to the formula,
    
           exp{-0.5 * ((x - m) @ cov ** -1 @ (x - m).T) } 
    MVN = -----------------------------------------------
              (2 * math.pi) ** 4 * det(cov)) ** 0.5
    
    exp: math.e raise to the power {param}
     x : feature vector
     m : mean vector
    cov: covariance matrix
    det: determinant
     @ : matrix multiplication
    '''
    # Correcting shapes for matmul compatibility
    _x = x.reshape((1, 4))
    _m = m.reshape((1, 4))
    
    # _a is the 1 /((2*pi) ** 4)
    _a = (2 * math.pi) ** 4
    
    # _b is the determinant of the covariance matrix
    _b = np.linalg.det(c)
    
    # _c is the total denominator of the expression
    _c = (_a * _b) ** (0.5)
    
    # _d is the (x - mean) @ covariance matrix ** -1 @ (x - mean).T
    _d = ((_x - _m) @ np.linalg.inv(c) @ (_x - _m).T)[0, 0]
    
    # _e is math.e raise to the power (-1 * (_d / 2)) (which is a scalar)
    _e = math.e ** (-_d/2)
    
    return _e / _c

# Cell 23
mvn_table = dict()

for k in range(1,4):
    mvn_table[k] = []
    for i in range(1,4):
        _mvn = mvn(xs[i], means[k], covariances[k])
        mvn_table[k].append(_mvn)

# print(f'| k | L(x_1; m_k, C_k) | L(x_2; m_k, C_k) | L(x_3; m_k, C_k) |')
# for key, value in mvn_table.items():
#     print(f'| {key}', *value, sep=' | ', end=' |\n')

    
print('Pretty Printed\n')
print(f'| k | L(x_1; m_k, C_k) | L(x_2; m_k, C_k) | L(x_3; m_k, C_k) |')
for key, value in mvn_table.items():
    _value = ' | '.join(map(lambda x: f'{x:.16f}'[:16], value))
    print(f'| {key}', _value, sep=' | ', end=' |\n')

