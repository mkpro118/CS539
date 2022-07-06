# Cell 1
# Part (a)
import numpy as np
import pandas as pd

file_name = 'iris.csv'

# A is the matrix represented by the csv file
A = pd.read_csv(file_name, header=None)

# For prettifying the DataFrame
label_map = dict([(i, (f'Feature {i+1}' if i != 4 else 'Label')) for i in range(5)])
A.rename(label_map, axis=1, inplace=True)

# The rows and columns of matrix A
rows, columns = A.shape
print(f'{rows = }', f'{columns = }')

# Cell 2
# Part (b)
# X is the feature matrix, y is the label vector
X, y = A[map(lambda x: f'Feature {x}', range(1,5))], A['Label']
X # Printing the feature matrix

# Cell 3
#  Part (b) continued
y # Printing the label vector

# Cell 4
# Part (c)

# The number of distinct label values
unique_label_count = y.nunique()

print(f'{unique_label_count = }')

# Cell 5
# Part (c) continued

number_of_occurences = y.value_counts()
print(number_of_occurences)

# Cell 6
# Part (d)

# For drawing the scatter plot
# %matplotlib inline

from matplotlib import pyplot as plt
plt.style.use('dark_background') # Set a dark theme for the plot

# Get the figure and axis to draw the scatter plot on
figure, axis = plt.subplots(figsize=(8,8,))

# Scatter plot with Feature 2 on the X-axis, Feature 3 on the Y-axis
axis.scatter(x=X['Feature 2'], y=X['Feature 3'], c=y, cmap='rainbow', linewidth=0.5, edgecolor='w')
axis.set_title('Feature 2 vs. Feature 3')
axis.set_xlabel('Feature 2')
axis.set_ylabel('Feature 3')

plt.show()

# Cell 7


