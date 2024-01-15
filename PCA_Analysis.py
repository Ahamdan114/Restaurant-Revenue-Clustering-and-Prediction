import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import StandardScaler

# Assuming your dataset is in a CSV file named 'your_dataset.csv'
file_path = r'D:\User\Ahmed\Country-data.csv'

# Specify columns to exclude
exclude_columns = ["country"]

# Set the limit to 10,000 rows
limit_rows = 10000

# Use pandas to read the CSV file into a DataFrame
df = pd.read_csv(file_path, nrows=limit_rows, usecols=lambda col: col not in exclude_columns)

print(df)

# Exclude specified columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
numerical_columns = [col for col in numerical_columns if col not in exclude_columns]

# Standardize the data
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Apply PCA to reduce dimensionality to 2 components: The goal is to emphasize parsimony
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df[numerical_columns])

# Plot the original and PCA-transformed data
plt.figure(figsize=(12, 4))

# Plot original data (considering only the first two features)
plt.subplot(1, 3, 1)
plt.scatter(df[numerical_columns].iloc[:, 0], df[numerical_columns].iloc[:, 1], marker='o')
plt.title('Original Data (First two features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot PCA-transformed data
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o')
plt.title('PCA Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Explained Variance
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratios:", explained_variance)

# Plot explained variance
plt.subplot(1, 3, 3)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance for each Principal Component')

plt.tight_layout()
plt.show()


# Calculate cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)

print(cumulative_variance)

plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')
plt.show()



# Access principal components
components = pca.components_

# Print loadings (correlation between original features and principal components)
print("Loadings:")
print(components)


eigenvalues = pca.explained_variance_
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.title('Scree Plot')
plt.show()
