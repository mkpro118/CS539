# Cell 4
# Question 2

# Setup
# %matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
plt.style.use('dark_background') # Styling the graphs

filename = 'iris.csv'

iris_details = np.genfromtxt(filename, delimiter=',') # loading the file

# features is a dictionary with feature numbers and corresponding vectors
# I'm using this dictionary to reduce code redundancy with 4 variables,
# while still being readable.
features = dict([(f'Feature {column+1}', iris_details[:, column]) for column in range(4)])

# label_vector is the vector of known classification labels
label_vector = iris_details[:,-1] # -1 since we know it's the last column

# Cell 5
# Question 2 part (a)
# Calculating the Sample Mean of the Features 

sample_means = dict([(key, np.mean(value)) for key, value in features.items()])

# Printing the sample mean for each feature
for feature, mean in sample_means.items():
    print(f'Sample mean for {feature} = {mean}\n')

# Cell 6
# Question 2 part (a) continued
# Calculating the Sample Standard Deviation of the Features

sample_std_dev = dict([(key, np.std(value)) for key, value in features.items()])

# Printing the sample standard deviation for each feature
for feature, std_dev in sample_std_dev.items():
    print(f'Sample standard deviation for {feature} = {std_dev}\n')

# Cell 7
# Question 2 part (b)
# The density=True option normalizes the histogram

# Get the figure and axes
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

# Plotting feature 1
ax1.hist(features['Feature 1'], bins=20, density=True, color='blue')
ax1.set_title('Feature 1')

# Plotting feature 2
ax2.hist(features['Feature 2'], bins=20, density=True, color='green')
ax2.set_title('Feature 2')

# Plotting feature 3
ax3.hist(features['Feature 3'], bins=20, density=True, color='red')
ax3.set_title('Feature 3')

# Plotting feature 4
ax4.hist(features['Feature 4'], bins=20, density=True, color='yellow')
ax4.set_title('Feature 4')

plt.show()

# Cell 8
# Question 2 part (c)
# Calculating the normalized feature vectors

# Function to normalize a given vector X
def normalize(X: np.ndarray) -> np.ndarray:
    '''
    Uses the given formula
    normalized(X) = (X - X.mean) / (X.std_dev)
    where X is the vector to normalize
    X.mean is the sample mean of the vector X
    X.std_dev is the sample standard deviation of the vector X
    '''
    return (X - np.mean(X)) / np.std(X)

normalized_features = dict([(key, normalize(value)) for key, value in features.items()])

# Cell 9
# Question 2 part (c) continued
# Calculating the sample mean of the normalized feature vectors

normalized_sample_means = dict([(key, np.mean(value)) for key, value in normalized_features.items()])

# Printing the sample mean for each normalized feature vector
for feature, mean in normalized_sample_means.items():
    print(f'Sample mean for {feature} = {mean}\n')

# Cell 10
# Question 2 part (c) continued
# Calculating the sample mean of the normalized feature vectors

normalized_sample_std_dev = dict([(key, np.std(value)) for key, value in normalized_features.items()])

# Printing the sample standard deviation for each normalized feature vector
for feature, std_dev in normalized_sample_std_dev.items():
    print(f'Sample mean for {feature} = {std_dev}\n')

# Cell 11
# Question 2 part (d)

# Get the figure and axes
fig2, ((_ax1, _ax2), (_ax3, _ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

# Plotting normalized feature 1
_ax1.hist(normalized_features['Feature 1'], bins=20, density=True, color='blue')
_ax1.set_title('Normalized Feature 1')

# Plotting normalized feature 2
_ax2.hist(normalized_features['Feature 2'], bins=20, density=True, color='green')
_ax2.set_title('Normalized Feature 2')

# Plotting normalized feature 3
_ax3.hist(normalized_features['Feature 3'], bins=20, density=True, color='red')
_ax3.set_title('Normalized Feature 3')

# Plotting normalized feature 4
_ax4.hist(normalized_features['Feature 4'], bins=20, density=True, color='yellow')
_ax4.set_title('Normalized Feature 4')

plt.show()

