import yfinance as yf
import pandas as pd

# define stocks we want to use
Tickers = ["NVDA", "GOOGL", "AAPL", "META"]

# download the data and store it in csv
df = yf.download(Tickers, start="2015-01-01", progress=False, auto_adjust=False)
df = df['Adj Close']

if isinstance(df.columns, pd.MultiIndex):
    prices = df['Adj Close'].copy()
else:
    prices = df.copy()

prices = prices[Tickers]
print("shape:", prices.shape)
print(prices.tail())
prices.to_csv("../data/prices_sample.csv")