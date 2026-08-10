import pandas as pd
import numpy as np

df = pd.read_csv("/home/jalcocert/Desktop/Py_RouteTracker/overlay/gforce_data.csv")

# Calculate Magnitude for each sample
df['mag'] = np.sqrt(df['c1']**2 + df['c2']**2 + df['c3']**2)

# Median Magnitude should represent 1G (Gravity)
one_g_raw = df['mag'].median()
print(f"Estimated 1G Raw Value: {one_g_raw:.2f}")

# Check which axis is dominant (Vertical)
mean_vals = df[['c1', 'c2', 'c3']].abs().mean()
print("Mean Absolute Values per Axis:")
print(mean_vals)
