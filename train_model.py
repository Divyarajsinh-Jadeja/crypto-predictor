# train_model.py

import pandas as pd
import numpy as np
import joblib  # ✅ Needed to save classifier
from xgboost import XGBClassifier
# At the bottom of train_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def prepare_lstm_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["close"]])
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

def train_lstm(df, symbol):
    X, y, scaler = prepare_lstm_data(df)
    if len(X) == 0: return

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{symbol}_lstm_model.h5")
    joblib.dump(scaler, f"models/{symbol}scaler.pkl")


# ✅ Train a binary classifier to predict whether price will go up
def train_classifier(df, symbol):
    df["target_class"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df[["close", "volume", "rsi", "ema7", "momentum"]]
    y = df["target_class"]
    split = int(len(X) * 0.8)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X[:split], y[:split])
    joblib.dump(model, f"models/{symbol}_classifier.pkl")  # ✅ Save model

# ✅ Fetch historical price data from Binance
def fetch_klines(symbol="BTCUSDT", interval="1d", limit=200):
    import requests
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    cols = ["timestamp", "open", "high", "low", "close", "volume", "*", "*", "*", "*", "*", "*"]
    df = pd.DataFrame(data, columns=cols).astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

# ✅ Add technical indicators
def add_features(df):
    df["rsi"] = compute_rsi(df["close"], 14)
    df["ema7"] = df["close"].ewm(span=7).mean()
    df["momentum"] = df["close"].diff()
    return df.dropna()

# ✅ RSI calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ✅ Estimate "confidence" as absolute % change
def compute_confidence(current, predicted):
    try:
        diff = predicted - current
        percent = (diff / current) * 100
        return round(abs(percent), 2)
    except:
        return 0
