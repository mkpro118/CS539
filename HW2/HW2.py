# Cell 6
# Question 2 part (d)

# Setup
# %matplotlib inline
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

plt.style.use('dark_background')

prob = np.array([0.05, 0.15, 0.40, 0.55, 0.25, 0.45, 0.48, 0.62, 0.67, 0.75])
y_true = np.array([0] * 4 + [1] * 6)

print(f'posterior probablities = {prob}')
print(f'actual labels = {y_true}')

# Cell 7
# Question 2 part (d) continued


def predict(pr, b):
    return (pr > b).astype(int)


def get_fpr_tpr(y_t, y_p):
    cm = confusion_matrix(y_t, y_p)
    fp, tp, fn, tn = cm[0, 1], cm[1, 1], cm[1, 0], cm[0, 0]
    fpr = fp / max(fp + tn, 1)
    tpr = tp / max(tp + fn, 1)
    return fpr, tpr


fpr_tpr_list = []

# list of fpr and tpr
for b in np.arange(0., 1., 0.05):
    y_pred = predict(prob, b)
    fpr, tpr = get_fpr_tpr(y_true, y_pred)
    fpr_tpr_list.append((fpr, tpr))

fpr_tpr = np.array(fpr_tpr_list)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.plot(fpr_tpr[:, 0], fpr_tpr[:, 1], linewidth=2)
ax.set_xlabel('False Positive Rate')
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_ylabel('True Positive Rate')
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_title('ROC Curve')

plt.show()

# Cell 8
# Question 2 part (d) continued


def roc_auc(fpr, tpr):
    indices = np.argsort(fpr)
    fpr = fpr[indices]
    tpr = tpr[indices]
    area = 0.
    for i in range(len(fpr) - 1):
        a = tpr[i]
        b = tpr[i + 1]
        h = fpr[i + 1] - fpr[i]
        area += 0.5 * h * (a + b)
    return area

# Calculating Area Under the ROC Curve
auc = roc_auc(fpr_tpr[:, 0], fpr_tpr[:, 1])

# Rounded to 5 decimals
print(f'{auc = :.5f}')

# Cell 9
# Question 3

# Setup
filename = 'mnist_test.csv'

mnist = np.genfromtxt(filename, delimiter=',')
features, labels = mnist[:, 1:], mnist[:, 0]

N = mnist.shape[0]
feature_dimension = mnist.shape[1] - 1

# part (a)
print(f'Number of samples {N = }')
print(f'Feature Dimension = {feature_dimension}')

# Cell 10
# Question 3 part (b)


def visualize(matrix):
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(12, 10))

    for i in range(20):
        r, c = divmod(i, 5)
        axs[r, c].imshow(np.reshape(matrix[i], (28, 28)), cmap=plt.get_cmap('gray'))
        axs[r, c].set_title(f'{int(labels[i])}')
        axs[r, c].set_xticks([])
        axs[r, c].set_yticks([])

    plt.show()

visualize(features)

# Cell 11
# Question 3 part (c)

X = features

U, S, Vh = np.linalg.svd(X, full_matrices=False)

S = np.log10(S)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

ax.plot(np.arange(*S.shape), S, linewidth=2)
ax.set_xlabel('Dimension')
ax.set_ylabel('Singular value')

plt.show()

# Cell 12
# Question 3 part (d)

V = Vh[:, :2]

Z = np.array([X[i] @ V for i in range(len(X))])

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))

label1 = labels == 1
label8 = labels == 8

ax1.scatter(Z[label1][:, 0], Z[label1][:, 1], c='hotpink')
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.set_title('Principal Components for label 1')

ax2.scatter(Z[label8][:, 0], Z[label8][:, 1], c='skyblue')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')
ax2.set_title('Principal Components for label 8')

plt.show()

# Cell 13
# Question 3 part (e)

Xh = Z @ V.T

visualize(Xh)

# Cell 14
# Question 4

# setup
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score

filename = 'winequality-red.csv'

data = np.genfromtxt(filename, delimiter=',')

# Label column was not specified, but inferred from data
X, y = data[1:, :-1], data[1:, -1]

print(f'{X.shape = }')
print(f'{y.shape = }')

# Cell 15
# Question 4, continued

# Divide training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(f'{X_train.shape = }')
print(f'{y_train.shape = }')
print(f'{X_test.shape = }')
print(f'{y_test.shape = }')

# Cell 16
# Question 4, continued

# Build a model
model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

# 3-Fold Cross Validation
cv3_scores = cross_val_score(model, X_train, y_train, cv=3)

print(f'Mean of Cross Validation scores: {np.mean(cv3_scores)}')
