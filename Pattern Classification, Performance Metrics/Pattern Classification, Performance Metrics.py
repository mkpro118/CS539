# Cell 2
# Question 2

# Setup
# %matplotlib inline
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
plt.style.use('dark_background')

filename = 'Ex_PC_metric.csv'

matrix = np.genfromtxt(filename, delimiter=',')

fig, ax = plt.subplots(1, 1, figsize=(6,6))

# ROC Curve
fpr, tpr, _ = roc_curve(matrix[:, 0], matrix[:, 1])
ax.plot(fpr, tpr)
ax.set_xlabel('FPR')
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_ylabel('TPR')
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_title('ROC Curve')

plt.show()

# Cell 3
# Question 2 continued

# AUC value
auc = roc_auc_score(matrix[:, 0], matrix[:, 1])

print(f'{auc = }')

