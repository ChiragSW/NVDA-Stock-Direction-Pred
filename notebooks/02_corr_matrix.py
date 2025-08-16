import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv("../data/prices_sample.csv", parse_dates=["Date"], index_col="Date")

# Calculate correlation matrix
corr_matrix = df.corr()

# Make sure plots directory exists
os.makedirs("../data/plots", exist_ok=True)

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Stock Price Correlation Matrix")
plt.savefig("../data/plots/correlation_matrix.png", dpi=300)
plt.close()

print(corr_matrix)
