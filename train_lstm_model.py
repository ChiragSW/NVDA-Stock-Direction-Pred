import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# define
DATA_PATH = "data/prices_sample.csv"
MODEL_DIR = "models"
PLOT_DIR = "data/plots"
LOOKBACK = 60
EPOCHS = 20
BATCH_SIZE = 32

# load data and once to sanity check
os.system("00_sanity_check.py")
df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
# print(df.shape)
# print(df.tail())

features = df[["NVDA", "GOOGL", "AAPL", "META"]].values

# scale features
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

features_scaled = features_scaled[:-1]
# create sequences
x, y = [], []
for i in range(LOOKBACK, len(features_scaled)):
    x.append(features_scaled[i-LOOKBACK:i, :])   # past 60 days, all 4 stocks
    y.append(features_scaled[i, 0])              # next day NVDA (col 0)

x, y = np.array(x), np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, x.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid') #output is the predicted NVDA price
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    x_train, y_train,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_data=(x_test, y_test),
    verbose=1
)

# save model and scalar
model.save(os.path.join(MODEL_DIR, "nvda_lstm_multi.h5"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "minmax_scaler_multi.pkl"))

print("MODEL and scalar saved")

# plot the training loss
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Training vs Validation Loss (Multi-stock LSTM)")
plt.savefig(os.path.join(PLOT_DIR, "loss_curve_multi.png"), dpi=300)
plt.close()