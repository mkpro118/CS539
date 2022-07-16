# Cell 1
# Question 1

# Setup
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from math import inf as infinity

filename = 'Ex_BVV_data.csv'

data = np.genfromtxt(filename, delimiter=',')

Xi = data[:, 0]
Yi = data[:, 1]
dXi = data[:, 2]

print(f'{Xi.shape = }')
print(f'{Yi.shape = }')
print(f'{dXi.shape = }')

# Cell 2
sample_size = 40
n_trials = 20
powers = range(1, 12)

def experiment(power):
    indices = np.random.permutation(Xi.shape[0])[:sample_size]
    fitted_polynomial = np.polyfit(Xi[indices], Yi[indices], power)
    activated = np.polyval(fitted_polynomial, Xi)
    mean = np.mean(activated)
    bias_squared = (np.linalg.norm(mean - dXi, 2) ** 2) / Xi.shape[0]
    variance = np.sum(np.var(activated)) / Xi.shape[0]
    return power, bias_squared + variance

def trial(power):
    _results = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(experiment, (power,) * n_trials)
        for result in results:
            _results.append(result)
    
    return min(_results, key=lambda x: x[0])

best_result = (0, infinity)

with ThreadPoolExecutor() as executor:
    results = executor.map(trial, powers)
    for order, min_b2_var in results:
        best_result = min((best_result, (order, min_b2_var)), key=lambda x: x[1])
        print(
            f"Order {order}: "
            f"Minimum Sum of Bias^2 and Variance = {min_b2_var:.5f}"
        )
print(f"Best result was with power {best_result[0]}, = {best_result[1]:.5f}")

