# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add Target Column
df['species'] = iris.target

# Display First 5 Rows
print(df.head())

# -----------------------------
# Features and Data Types
# -----------------------------
print("\nFeatures and Data Types:")
print(df.dtypes)

# -----------------------------
# Summary Statistics
# -----------------------------
print("\nSummary Statistics:")
print(df.describe())

# Additional Statistics
print("\nVariance:")
print(df.var(numeric_only=True))

print("\nRange:")
print(df.max(numeric_only=True) - df.min(numeric_only=True))

# -----------------------------
# Histogram for Each Feature
# -----------------------------
df.hist(figsize=(10,8))
plt.suptitle("Histogram of Iris Dataset Features")
plt.show()

# -----------------------------
# Boxplot for Each Feature
# -----------------------------
plt.figure(figsize=(10,6))
df.boxplot()
plt.title("Boxplot of Iris Dataset Features")
plt.show()

