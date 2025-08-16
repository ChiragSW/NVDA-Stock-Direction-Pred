import pandas as pd
import os

# Load tatic prices
df = pd.read_csv("../data/prices_sample.csv", parse_dates=["Date"], index_col="Date")

# daily returns
returns = df.pct_change().add_suffix("_return")

# lag features
lags = {}
for lag in [1, 2, 3]:  # 1-day, 2-day, 3-day lag
    lagged = df.shift(lag).add_suffix(f"_lag{lag}")
    lags[lag] = lagged

# moving averages
ma = {}
for window in [5, 10, 20]:
    moving_avg = df.rolling(window).mean().add_suffix(f"_ma{window}")
    ma[window] = moving_avg

# all features are combined
df_features = pd.concat([df, returns] + list(lags.values()) + list(ma.values()), axis=1)

df_features = df_features.dropna()

# Save processed features
df_features.to_csv("../data/processed/processed_features.csv")

print("Shape:", df_features.shape)
print(df_features.head())
