import numpy as np
from sklearn.datasets import load_iris

# Load the IRIS dataset
data = load_iris()

# Print the feature names
print("Feature names:", data.feature_names)

# Print the target names
print("Target names:", data.target_names)

# Print the first 5 samples
print("First 5 samples:")
print(data.data[:5])
print(data.target[:5])
