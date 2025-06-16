# ✅ train_20_lstm.py — Train LSTM for Top 20 Coins
import os
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Binance Kline Fetcher ---
def fetch_klines(symbol="BTCUSDT", interval="1h", limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=10).json()
        df = pd.DataFrame(res, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df[["close", "volume"]]
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()

# --- Feature Engineering ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    df["rsi"] = compute_rsi(df["close"], 14)
    df["ema7"] = df["close"].ewm(span=7).mean()
    df["momentum"] = df["close"].diff()
    return df.dropna()

# --- LSTM Data Prep ---
def prepare_lstm_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["close"]])
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i - 60:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

# --- Training Function ---
def train_lstm(symbol, coin_id):
    df = fetch_klines(symbol)
    if df.empty:
        print(f"❌ Skipped {symbol} — No data")
        return
    df = add_features(df)
    if df.empty or len(df) < 100:
        print(f"⚠️ Skipped {symbol} — Not enough feature data")
        return

    X, y, scaler = prepare_lstm_data(df)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{symbol}_lstm_model.h5")
    joblib.dump(scaler, f"models/{symbol}scaler.pkl")
    print(f"✅ {coin_id} ({symbol}) model saved.")

# --- Top 20 Coins (CoinGecko ID to Binance Symbol) ---
TOP_20 = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
    "ripple": "XRPUSDT",
    "cardano": "ADAUSDT",
    "dogecoin": "DOGEUSDT",
    "avalanche": "AVAXUSDT",
    "shiba-inu": "SHIBUSDT",
    "polkadot": "DOTUSDT",
    "polygon": "MATICUSDT",
    "tron": "TRXUSDT",
    "chainlink": "LINKUSDT",
    "litecoin": "LTCUSDT",
    "uniswap": "UNIUSDT",
    "bitcoin-cash": "BCHUSDT",
    "stellar": "XLMUSDT",
    "near": "NEARUSDT",
    "internet-computer": "ICPUSDT",
    "aptos": "APTUSDT"
}

# --- Main ---
if __name__ == "__main__":
    for coin_id, symbol in TOP_20.items():
        train_lstm(symbol, coin_id)
