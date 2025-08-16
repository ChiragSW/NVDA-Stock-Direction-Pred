import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load processed features (all four tickers + engineered features if any)
df = pd.read_csv("../data/processed/processed_features.csv", parse_dates=["Date"], index_col="Date")

df = df.sort_index()

# Target: NVDA closing price
target_col = "NVDA"
y = df[target_col].values

# Features: everything else
x = df.drop(columns=[target_col]).values

# Scale features for model stability
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, shuffle=False)

# Sequence creation function for fixed window
def create_sequences(X, y, time_steps=60):  # 60-day lookback
    xs, ys = [], []
    for i in range(len(X) - time_steps):
        xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(xs), np.array(ys)

# Create sequences
time_steps = 60
x_train_seq, y_train_seq = create_sequences(x_train, y_train, time_steps)
x_test_seq, y_test_seq = create_sequences(x_test, y_test, time_steps)

# Shapes for confirmation
print("x_train_seq shape:", x_train_seq.shape)  # (samples, 60, features)
print("y_train_seq shape:", y_train_seq.shape)  # (samples,)
print("x_test_seq shape:", x_test_seq.shape)
print("y_test_seq shape:", y_test_seq.shape)

# Save sequences for Phase 4 model training
np.save("../data/processed/x_train_seq.npy", x_train_seq)
np.save("../data/processed/y_train_seq.npy", y_train_seq)
np.save("../data/processed/x_test_seq.npy", x_test_seq)
np.save("../data/processed/y_test_seq.npy", y_test_seq)

print("✅ Phase 3.2 complete — Sequences saved for Phase 4.")
