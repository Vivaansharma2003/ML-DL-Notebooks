import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("/Users/vivaansharma/Desktop/ML-DL-Notebooks/16th_December/CCPP Data.csv")

# Reproducibility
np.random.seed(42)

# Very small fraction of missing values
missing_fraction = 0.03  # 3%

# Random mask
mask = np.random.rand(*df.shape) < missing_fraction

# Apply missing values
df_missing = df.mask(mask)

# Save to new CSV
df_missing.to_csv("CCPP_Data.csv", index=False)

print("Original Data:")
print(df)

print("\nData with Missing Values (saved as data_with_missing_values.csv):")
print(df_missing)
