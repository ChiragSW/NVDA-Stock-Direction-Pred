import pandas as pd
import matplotlib.pyplot as plt

# Load the saved CSV
df = pd.read_csv("../data/prices_sample.csv", parse_dates=["Date"], index_col="Date")

# Plot
plt.figure(figsize=(12, 6))
for ticker in df.columns:
    plt.plot(df.index, df[ticker], label=ticker)

plt.title("Stock Prices of NVDA, GOOGL, AAPL, META")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price (USD)")
plt.legend()
plt.grid(True)
plt.savefig("../data/plots/raw_prices.png", dpi = 300)
plt.show()

# normalize to see progress in %
(df / df.iloc[0]).plot(figsize=(12,6))
plt.title("Normalized Stock Price Comparison")
plt.xlabel("Date")
plt.ylabel("Price (Normalized to 1.0)")
plt.grid(True)
plt.savefig("../data/plots/normalized_prices.png", dpi = 300)
plt.show()
