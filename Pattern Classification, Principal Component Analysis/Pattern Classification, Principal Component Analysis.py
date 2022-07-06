# Cell 2
# Question 2

# Setup
# %matplotlib inline
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('dark_background')

filename = 'EX_PCAdata.csv'

matrix = np.genfromtxt(filename, delimiter=',')
features = matrix[:, :-1]
labels = matrix[:, -1]

pca = PCA(n_components=2)

transformed = pca.fit(features).transform(features)
t1 = transformed[labels == 4]
t2 = transformed[labels == 7]

fig, ax = plt.subplots(1, 1, figsize=(8,8))

ax.scatter(t1[:, 0], t1[:, 1], c='lightblue')
ax.scatter(t2[:, 0], t2[:, 1], marker='^', c='hotpink')

plt.show()

