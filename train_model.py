# ✅ Updated train_model.py with fixed fetch_klines call in train_all_models

import pandas as pd
import numpy as np
import joblib
import os
import requests
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import pickle
import warnings
import time
from datetime import datetime, timedelta
from db_manager import save_historical_data_to_db  # Import caching function

warnings.filterwarnings('ignore')

COIN_ID_MAP = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin",
    "XRPUSDT": "ripple",
    "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin",
    "AVAXUSDT": "avalanche",
    "SHIBUSDT": "shiba-inu",
    "DOTUSDT": "polkadot",
    "MATICUSDT": "polygon",
    "TRXUSDT": "tron",
    "LINKUSDT": "chainlink",
    "LTCUSDT": "litecoin",
    "UNIUSDT": "uniswap",
    "BCHUSDT": "bitcoin-cash",
    "XLMUSDT": "stellar",
    "NEARUSDT": "near",
    "ICPUSDT": "internet-computer",
    "APTUSDT": "aptos"
}

# === Feature Engineering ===
def compute_rsi(series, period=14):
    """Compute Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, period=20, std_dev=2):
    """Compute Bollinger Bands"""
    rolling_mean = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band

def compute_macd(series, fast=12, slow=26, signal=9):
    """Compute MACD indicator"""
    exp1 = series.ewm(span=fast).mean()
    exp2 = series.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def add_features(df):
    """Add technical indicators to the dataframe"""
    df["rsi"] = compute_rsi(df["close"], 14)
    df["ema7"] = df["close"].ewm(span=7).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["momentum"] = df["close"].diff()
    df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["close"])
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["macd"], df["macd_signal"], df["macd_histogram"] = compute_macd(df["close"])
    df["price_change"] = df["close"].pct_change()
    df["volatility"] = df["close"].rolling(10).std()
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]
    df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]
    df["co_ratio"] = (df["close"] - df["open"]) / df["open"]
    return df.dropna()

# === LSTM Training ===
def prepare_lstm_data(df, lookback=60):
    """Prepare data for LSTM training"""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["close"]])
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

def train_lstm(df, symbol, lookback=60, epochs=20):
    """Train LSTM model for price prediction"""
    try:
        X, y, scaler = prepare_lstm_data(df, lookback)
        if len(X) == 0:
            print(f"⚠️ Not enough data for LSTM training for {symbol}")
            return False
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            LSTM(50, return_sequences=True),
            LSTM(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        os.makedirs("models", exist_ok=True)
        model.save(f"models/{symbol}_lstm_model.h5")
        joblib.dump(scaler, f"models/{symbol}scaler.pkl")
        print(f"✅ LSTM model trained for {symbol}")
        return True
    except Exception as e:
        print(f"❌ LSTM training failed for {symbol}: {e}")
        return False

def train_classifier(df, symbol):
    """Train XGBoost classifier for price direction prediction"""
    try:
        df["target_class"] = (df["close"].shift(-1) > df["close"]).astype(int)
        feature_cols = [
            "close", "volume", "rsi", "ema7", "ema21", "momentum",
            "bb_position", "macd", "macd_histogram", "volatility",
            "volume_ratio", "hl_ratio", "co_ratio"
        ]
        X = df[feature_cols].fillna(0)
        y = df["target_class"]
        X = X[:-1]
        y = y[:-1]
        if len(X) == 0:
            print(f"⚠️ Not enough data for classifier training for {symbol}")
            return False
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{symbol}_classifier.pkl")
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"✅ Classifier trained for {symbol} - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        return True
    except Exception as e:
        print(f"❌ Classifier training failed for {symbol}: {e}")
        return False

def train_prophet(df, symbol):
    """Train Prophet model for time series forecasting"""
    try:
        df_prophet = df[["timestamp", "close"]].copy()
        df_prophet["ds"] = pd.to_datetime(df_prophet["timestamp"], unit="ms")
        df_prophet["y"] = df_prophet["close"]
        if len(df_prophet) < 30:
            print(f"⚠️ Not enough data for Prophet training for {symbol}")
            return False
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(df_prophet[["ds", "y"]])
        os.makedirs("models", exist_ok=True)
        with open(f"models/{symbol}_prophet.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Prophet model trained for {symbol}")
        return True
    except Exception as e:
        print(f"❌ Prophet training failed for {symbol}: {e}")
        return False

def fetch_klines(symbol="BTCUSDT", interval="1d", years=None, days=None, batch_limit=1000, max_retries=3):
    """Fetch historical price data from Binance API
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Kline interval (default: "1d")
        years: Number of years of data to fetch (default: 3 if days not specified)
        days: Number of days of data to fetch (takes precedence over years)
        batch_limit: Maximum klines per API call (default: 1000)
        max_retries: Maximum retry attempts (default: 3)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        url = f"https://api.binance.com/api/v3/klines"
        end_time = int(time.time() * 1000)  # Current timestamp in ms
        
        # Calculate start time based on days or years
        if days is not None:
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            time_desc = f"{days} days"
        elif years is not None:
            start_time = int((datetime.now() - timedelta(days=years*365)).timestamp() * 1000)
            time_desc = f"{years} years"
        else:
            # Default to 3 years (minimum required for training) if neither specified
            start_time = int((datetime.now() - timedelta(days=3*365)).timestamp() * 1000)
            time_desc = "3 years"
        
        all_data = []
        batch_count = 0
        data = None  # Initialize to avoid NameError

        print(f"📡 Fetching {time_desc} of data for {symbol} in batches...")

        while start_time < end_time:
            retries = 0
            batch_success = False
            while retries < max_retries:
                try:
                    params = {
                        "symbol": symbol,
                        "interval": interval,
                        "limit": batch_limit,
                        "startTime": start_time,
                        "endTime": end_time
                    }
                    response = requests.get(url, params=params, timeout=30)  # Increased timeout
                    response.raise_for_status()
                    data = response.json()
                    if not data:
                        print(f"ℹ️ No more data available for {symbol}")
                        batch_success = True
                        break
                    all_data.extend(data)
                    start_time = int(data[-1][0]) + 1
                    batch_count += 1
                    print(f"✅ Fetched batch {batch_count} with {len(data)} klines")
                    time.sleep(0.2)  # Avoid rate limiting
                    batch_success = True
                    break
                except requests.exceptions.Timeout:
                    retries += 1
                    if retries == max_retries:
                        print(f"❌ Timeout fetching batch for {symbol} after {max_retries} retries")
                        # Return partial data if we have at least 60 rows
                        if len(all_data) >= 60:
                            print(f"⚠️ Returning partial data ({len(all_data)} rows) due to timeout")
                            break
                        return None
                    print(f"⚠️ Retry {retries}/{max_retries} for {symbol}: Timeout")
                    time.sleep(2 ** retries)  # Exponential backoff
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        print(f"❌ Failed to fetch batch for {symbol} after {max_retries} retries: {e}")
                        # Return partial data if we have at least 60 rows
                        if len(all_data) >= 60:
                            print(f"⚠️ Returning partial data ({len(all_data)} rows) due to error")
                            break
                        return None
                    print(f"⚠️ Retry {retries}/{max_retries} for {symbol}: {e}")
                    time.sleep(2 ** retries)  # Exponential backoff

            if not batch_success or not data:
                break

        if not all_data:
            print(f"❌ No data fetched for {symbol}")
            return None

        cols = ["timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"]
        df = pd.DataFrame(all_data, columns=cols)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["timestamp"] = df["timestamp"].astype(int)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        if days:
            print(f"✅ Total {len(df)} klines fetched for {symbol} (~{len(df)} days)")
        else:
            print(f"✅ Total {len(df)} klines fetched for {symbol} (~{len(df)/365:.1f} years)")
        return df

    except Exception as e:
        print(f"❌ Failed to fetch data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_confidence(current, predicted):
    """Compute confidence score based on price difference"""
    try:
        if current <= 0:
            return 0
        diff = predicted - current
        percent = abs(diff / current) * 100
        return round(min(percent, 100), 2)
    except:
        return 0

def train_all_models():
    """Train all models for all supported cryptocurrencies with minimum 3 years of data"""
    print("🚀 Starting model training for all cryptocurrencies...")
    print("📊 Minimum requirement: 3 years (1095 days) of historical data")
    
    MIN_TRAINING_DAYS = 3 * 365  # Minimum 3 years = 1095 days
    
    for symbol, coin_id in COIN_ID_MAP.items():
        print(f"\n📊 Processing {symbol} ({coin_id})...")
        
        # Fetch minimum 3 years of data for training
        df = fetch_klines(symbol, years=3)
        
        if df is None:
            print(f"❌ Failed to fetch data for {symbol}, skipping...")
            continue
        
        # Validate minimum 3 years of data
        if len(df) < MIN_TRAINING_DAYS:
            print(f"⚠️ Insufficient data for {symbol}: {len(df)} days (minimum {MIN_TRAINING_DAYS} days required), skipping...")
            continue
        
        print(f"✅ Fetched {len(df)} days (~{len(df)/365:.1f} years) of data for {symbol}")
        
        # ✅ Save fetched data to MongoDB cache for future predictions
        print(f"💾 Saving {symbol} data to MongoDB cache...")
        save_historical_data_to_db(symbol, df)
        
        df = add_features(df)
        if len(df) < 60:
            print(f"⚠️ Not enough data after feature engineering for {symbol}, skipping...")
            continue
        
        lstm_success = train_lstm(df, symbol)
        classifier_success = train_classifier(df, symbol)
        prophet_success = train_prophet(df, symbol)
        success_count = sum([lstm_success, classifier_success, prophet_success])
        print(f"📈 {symbol} completed: {success_count}/3 models trained successfully")

if __name__ == "__main__":
    train_all_models()
    print("\n🎉 Model training completed!")