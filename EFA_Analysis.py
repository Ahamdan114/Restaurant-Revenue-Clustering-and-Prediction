import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

# Assuming your dataset is in a CSV file named 'your_dataset.csv'
file_path = r'D:\User\Ahmed\Country-data.csv'

# Specify columns to exclude
exclude_columns = ["country"]

# Set the limit to 10,000 rows
limit_rows = 10000

# Use pandas to read the CSV file into a DataFrame
df = pd.read_csv(file_path, nrows=limit_rows, usecols=lambda col: col not in exclude_columns)

# Instantiate the FactorAnalyzer with the number of factors you want to extract
n_factors = 3
fa = FactorAnalyzer(n_factors, rotation='varimax')

# Fit the model to the data
fa.fit(df)

# Get factor loadings
factor_loadings = fa.loadings_

# Print factor loadings
print("Factor Loadings:")
print(factor_loadings)

# Scree plot to determine the number of factors
ev, v = fa.get_eigenvalues()
plt.scatter(range(1, len(df.columns) + 1), ev)
plt.plot(range(1, len(df.columns) + 1), ev)
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
