import os
import numpy as np
import pandas as pd 
from tensorflow.keras.models import load_model
import joblib
from send_email import send_email

# load data
df = pd.read_csv("data/prices_sample.csv", index_col="Date", parse_dates=True)
features = df[["NVDA", "GOOGL", "AAPL", "META"]].values

# load scaler and transforms
scaler = joblib.load("models/minmax_scaler_multi.pkl")
features_scaled = scaler.transform(features)

# prepare the last 60 day window
last_window = features_scaled[-60:]  # shape (60, 4)
x_input = np.expand_dims(last_window, axis=0)  # shape (1, 60, 4)

# load model and predict
model = load_model("models/nvda_lstm_multi.h5")
prob_up = model.predict(x_input)[0][0]

print("Last Date in Data:", df.index[-1].date())
print(f"Last NVDA stock Closed at: {df['NVDA'].iloc[-1]:.2f}")
print(f"Probability that NVDA goes UP: {(prob_up * 100):.2f}%")

if prob_up > 0.5:
    print("Model predicts: NVDA will go UP ^^^^")
else:
    print("Model predicts: NVDA will go DOWN !!!!")

result = "UP" if prob_up > 0.5 else "DOWN"

subject = f"NVDA Prediction: {result} ({prob_up:.2f} confidence)"
body = f"""
Date: {df.index[-1].date()}
Last Close: {df['NVDA'].iloc[-1]:.2f}

Greetings Master!
Model Prediction for NVDA stock:
NVDA likely will go {result} tomorrow
Confidence: {prob_up:.2f}
"""

send_email(subject, body)