# Cell 1
# Setup
import numpy as np

filename = 'Ex_PC_data.csv'

# Cell 2
# Question 1 part (a)

# Reading the data file and converting it into a matrix.
data = np.genfromtxt(filename, delimiter=',')
print(data)


# Since we're given that the last column is labels
features, labels = data[:, :-1], data[:, -1].astype(int)

# Cell 3
# Question 1 part (b)

# Number of samples and features
n_samples, n_features = features.shape

print(f'Number of samples = {n_samples}')
print(f'Number of features = {n_features}')

# Cell 4
# Question 1 part (b) continued

classes = np.unique(labels)
n_classes = classes.shape[0]

print(f'Number of classes = {n_classes}')
print(f'The classes are:', ', '.join(map(str, classes)))

# Cell 5
# Question 1 part (b) continued

for i in classes:
    # Number of labels with class i
    count = np.sum(labels == i)
    
    # pdf of that class
    pdf = count / n_samples
    
    # rounded to 5 decimal places
    print(f'Class distribution for label {i} = {pdf:.5}')
    

# Cell 6
# Question 1 part (c)

# To impute, we need to first calculate the means of the features
feature_means = np.nanmean(features, axis=0)
print(f'Mean Vector for the features = \n{feature_means}')

# Cell 7
# Question 1 part (c) continued

# Imputing nan values,
for i, m in enumerate(feature_means):
    # determining where the nan's are in each column
    nans = np.where(np.isnan(features[:,i]))[0]
    for j in nans:
        features[j, i] = m
print(features)

# Cell 8
# Question 1 part (d)

ratio = 0.8 # 80 / 20 split

# Number of elements in the training set
length = int(features.shape[0] * ratio)

# get a random permutation of the indices.
indices = np.random.permutation(features.shape[0])

train_features = features[indices[:length]]
train_labels   = labels[indices[:length]]
test_features  = features[indices[length:]]
test_labels    = labels[indices[length:]]

print("Shape of training dataset's features =", train_features.shape)
print("Shape of training dataset's labels   =", train_labels.shape)
print("Shape of testing dataset's features  =", test_features.shape)
print("Shape of testing dataset's labels    =", test_labels.shape)

# Cell 9
# Question 1 part (e)

max_vector = np.max(train_features, axis=0)
min_vector = np.min(train_features, axis=0)
print(max_vector)
print(min_vector)

# Cell 10
# Question 1 part (e) continued

def scale(X: np.ndarray, start: float, end: float) -> np.ndarray:
    return (end - start) * (X - min_vector) / (max_vector - min_vector) + start

scaled_training_features = scale(train_features, -5, 5)
# print(f'scaled_training_features\n{scaled_training_features}')

scaled_testing_features = scale(test_features, -5, 5)
# print(f'scaled_testing_features\n{scaled_testing_features}')

print('Max in scaled testing,', np.max(scaled_testing_features))
print('Min in scaled testing,', np.min(scaled_testing_features))

# Cell 11
# Question 1 part (f)

mean_scaled = np.mean(scaled_training_features, axis=0)
std_scaled = np.std(scaled_training_features, axis=0)

def normalize(X: np.ndarray) -> np.ndarray:
    return (X - mean_scaled) / std_scaled

n_s_train = normalize(scaled_training_features)
n_s_test = normalize(scaled_testing_features)

# print('Scaled and Normalized Training Features,', n_s_train, sep='\n')
# print('Scaled and Normalized Testing Features,', n_s_test, sep='\n')

_mean = np.mean(n_s_test, axis=0)
print('Testing Features mean, ', _mean)
_std = np.std(n_s_test, axis=0)
print('Testing Features std, ', _std)

# Cell 12
# Question 1 part (g)

from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
for train_index, test_index in kf.split(n_s_train):
    X_train = n_s_train[train_index]
    y_train = train_labels[train_index]

for i in classes:
    # Number of labels with class i
    count = np.sum(y_train == i)
    
    # pdf of that class
    pdf = count / y_train.shape[0]
    
    # rounded to 5 decimal places
    print(f'Class distribution for label {i} = {pdf:.5}')

