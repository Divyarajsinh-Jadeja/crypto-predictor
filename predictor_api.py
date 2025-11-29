import os
import pickle
from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd

# Suppress TensorFlow warnings and CPU instruction messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model
from train_model import add_features, compute_confidence, fetch_klines
import warnings
from db_manager import (
    log_prediction_to_db,
    get_historical_data_from_db,
    get_last_cached_timestamp,
    update_recent_data_in_cache
)
from sentiment_analyzer import get_crypto_sentiment, adjust_confidence_with_sentiment
import time
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

app = Flask(__name__)

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

SYMBOL_MAP = {
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


def check_model_files(symbol):
    """Check if required model files exist"""
    required_files = [
        f"models/{symbol}_lstm_model.h5",
        f"models/{symbol}scaler.pkl"
    ]

    optional_files = [
        f"models/{symbol}_classifier.pkl",
        f"models/{symbol}_prophet.pkl"
    ]

    missing_required = [f for f in required_files if not os.path.exists(f)]
    available_optional = [f for f in optional_files if os.path.exists(f)]

    return len(missing_required) == 0, available_optional


def get_lstm_prediction(df, symbol):
    """Get LSTM model prediction"""
    try:
        model = load_model(f"models/{symbol}_lstm_model.h5")
        scaler = joblib.load(f"models/{symbol}scaler.pkl")

        # Prepare input data (last 60 data points)
        X_pred = df[["close"]].values[-60:]
        if len(X_pred) < 60:
            print(f"⚠️ Not enough data for LSTM prediction for {symbol}")
            return None

        X_pred_scaled = scaler.transform(X_pred)
        X_pred_scaled = np.reshape(X_pred_scaled, (1, X_pred_scaled.shape[0], 1))

        # Make prediction
        pred_scaled = model.predict(X_pred_scaled, verbose=0)[0][0]
        predicted_price = scaler.inverse_transform([[pred_scaled]])[0][0]

        return float(predicted_price)

    except Exception as e:
        print(f"❌ LSTM prediction failed for {symbol}: {e}")
        return None


def get_prophet_prediction(symbol):
    """Get Prophet model prediction"""
    try:
        with open(f"models/{symbol}_prophet.pkl", "rb") as f:
            prophet = pickle.load(f)

        # Create future dataframe for next day
        future = pd.DataFrame([{
            "ds": pd.Timestamp.now() + pd.Timedelta(days=1)
        }])

        forecast = prophet.predict(future)
        prophet_pred = forecast["yhat"].values[0]

        return float(prophet_pred)

    except Exception as e:
        print(f"❌ Prophet prediction failed for {symbol}: {e}")
        return None


def get_classifier_prediction(df, symbol):
    """Get classifier prediction (success probability)"""
    try:
        clf = joblib.load(f"models/{symbol}_classifier.pkl")

        # Prepare features (using the same features as in training)
        feature_cols = [
            "close", "volume", "rsi", "ema7", "ema21", "momentum",
            "bb_position", "macd", "macd_histogram", "volatility",
            "volume_ratio", "hl_ratio", "co_ratio"
        ]

        # Get the latest features
        X_latest = df[feature_cols].iloc[-1:].copy()

        # Handle any missing values
        X_latest = X_latest.fillna(0)

        # Get prediction probability
        proba = clf.predict_proba(X_latest)[0][1]  # Probability of price going up
        success_chance = float(round(proba * 100, 2))

        return success_chance

    except Exception as e:
        print(f"❌ Classifier prediction failed for {symbol}: {e}")
        return None


@app.route("/predict_all_lstm", methods=["GET"])
def predict_all_lstm():
    """Get predictions for all cryptocurrencies"""
    results = []
    coins = list(SYMBOL_MAP.items())
    total = len(coins)
    
    for idx, (coin, symbol) in enumerate(coins, 1):
        try:
            print(f"📊 Processing {coin} ({idx}/{total})...")
            result = run_prediction(symbol)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {coin}: {e}")
            results.append({"symbol": symbol, "error": str(e)})
    
    return jsonify(results)


@app.route("/predict_batch", methods=["GET"])
def predict_batch():
    """Get predictions for a batch of coins (to avoid timeout)"""
    coins_param = request.args.get("coins", "")
    if not coins_param:
        return jsonify({"error": "Please provide coins parameter (comma-separated)"}), 400
    
    coins_list = [c.strip().lower() for c in coins_param.split(",")]
    results = []
    
    for coin in coins_list:
        if coin not in SYMBOL_MAP:
            results.append({"error": f"Unsupported coin: {coin}"})
            continue
        
        symbol = SYMBOL_MAP[coin]
        result = run_prediction(symbol)
        results.append(result)
    
    return jsonify(results)


@app.route("/predict", methods=["GET"])
def predict():
    coin = request.args.get("coin", "bitcoin")

    if coin not in SYMBOL_MAP:
        return jsonify({
            "error": "Unsupported coin",
            "supported_coins": list(SYMBOL_MAP.keys())
        }), 400

    symbol = SYMBOL_MAP[coin]
    result = run_prediction(symbol)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


@app.route("/send-to-chat", methods=["POST", "GET"])
def send_to_chat():
    """Send all predictions to Google Chat"""
    try:
        from gchat_bot import send_all_predictions
        send_all_predictions()
        return jsonify({
            "status": "success",
            "message": "Predictions sent to Google Chat successfully"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to send predictions: {str(e)}"
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "API is live"}), 200


@app.route("/health/db", methods=["GET"])
def db_health_check():
    """Check MongoDB connection and cache status"""
    from db_manager import verify_mongodb_connection
    status = verify_mongodb_connection()
    if status.get("connected"):
        return jsonify({
            "status": "ok",
            "database": status.get("database"),
            "collections": status.get("collections"),
            "database_size_mb": status.get("database_size_mb"),
            "storage_size_mb": status.get("storage_size_mb")
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": status.get("error", "MongoDB connection failed")
        }), 503


@app.route("/models/status", methods=["GET"])
def model_status():
    status = {}
    for coin, symbol in SYMBOL_MAP.items():
        try:
            lstm_model = f"models/{symbol}_lstm_model.h5"
            clf_model = f"models/{symbol}_classifier.pkl"
            scaler = f"models/{symbol}scaler.pkl"
            all_exist = all(os.path.exists(p) for p in [lstm_model, clf_model, scaler])
            status[coin] = "✅ Ready" if all_exist else "❌ Missing files"
        except:
            status[coin] = "❌ Error checking files"
    return jsonify(status)

def run_prediction(symbol, live_price=None):
    """Reusable prediction logic for WebSocket or batch jobs with MongoDB caching"""
    try:
        has_required, optional_models = check_model_files(symbol)
        if not has_required:
            return {"symbol": symbol, "error": "Required model files not found"}

        # ✅ Step 1: Try to get data from MongoDB cache first
        print(f"📦 Checking MongoDB cache for {symbol}...")
        df = get_historical_data_from_db(symbol)
        
        # ✅ Step 2: Check if cache is recent (within last 2 days)
        cache_is_recent = False
        if df is not None and len(df) > 0:
            last_timestamp = get_last_cached_timestamp(symbol)
            if last_timestamp:
                # Check if last cached data is within last 2 days
                two_days_ago = int((datetime.now() - timedelta(days=2)).timestamp() * 1000)
                cache_is_recent = last_timestamp >= two_days_ago
                
                if cache_is_recent:
                    print(f"✅ Using cached data for {symbol} ({len(df)} records, last update: {datetime.fromtimestamp(last_timestamp/1000)})")
                else:
                    print(f"⚠️ Cache for {symbol} is outdated (last update: {datetime.fromtimestamp(last_timestamp/1000)}), fetching recent data...")
        
        # ✅ Step 3: If cache is missing or old, fetch only last 2 days from Binance
        if df is None or not cache_is_recent:
            print(f"📡 Fetching last 2 days of data for {symbol} from Binance...")
            recent_df = fetch_klines(symbol, days=2)
            
            if recent_df is None or len(recent_df) == 0:
                # If we have cached data, use it even if it's old
                if df is not None and len(df) >= 60:
                    print(f"⚠️ Failed to fetch recent data, using cached data for {symbol}")
                else:
                    # No cache and fetch failed - try fetching more days as fallback
                    print(f"⚠️ Cache unavailable and recent fetch failed, trying 200 days fallback...")
                    df = fetch_klines(symbol, days=200)
                    if df is None or len(df) < 60:
                        return {"symbol": symbol, "error": "Insufficient data"}
                # Skip to feature engineering
            else:
                # ✅ Step 4: Merge recent data with cached data
                if df is not None and len(df) > 0:
                    # Remove duplicates and merge
                    print(f"🔄 Merging {len(recent_df)} recent records with {len(df)} cached records...")
                    # Combine dataframes
                    combined_df = pd.concat([df, recent_df], ignore_index=True)
                    # Remove duplicates based on timestamp
                    combined_df = combined_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                    df = combined_df
                    print(f"✅ Combined data: {len(df)} total records")
                    
                    # ✅ Step 5: Update cache with recent data
                    update_recent_data_in_cache(symbol, recent_df)
                else:
                    # No cache exists, use recent data and save it
                    df = recent_df
                    print(f"💾 Saving {len(df)} records to cache (first time)...")
                    from db_manager import save_historical_data_to_db
                    save_historical_data_to_db(symbol, df)
        
        # ✅ Step 6: Validate we have enough data
        if df is None or len(df) < 60:
            return {"symbol": symbol, "error": "Insufficient data"}

        df = add_features(df)
        current = float(df["close"].iloc[-1])
        lstm_pred = get_lstm_prediction(df, symbol)
        prophet_pred = get_prophet_prediction(symbol) if f"models/{symbol}_prophet.pkl" in optional_models else None
        classifier_prob = get_classifier_prediction(df, symbol) if f"models/{symbol}_classifier.pkl" in optional_models else None

        if prophet_pred:
            final_pred = 0.6 * lstm_pred + 0.4 * prophet_pred
        else:
            final_pred = lstm_pred

        signal = "Buy" if final_pred > current else "Sell"
        confidence = compute_confidence(current, final_pred)
        
        # ✅ Sentiment Analysis Integration
        coin_name = COIN_ID_MAP.get(symbol, "bitcoin")
        sentiment_score = get_crypto_sentiment(coin_name)
        final_confidence = adjust_confidence_with_sentiment(confidence, sentiment_score)

        # ✅ Log prediction to MongoDB
        log_prediction_to_db(
            symbol=symbol, 
            signal=signal, 
            price=final_pred, 
            confidence=final_confidence,
            success_chance=classifier_prob
        )

        return {
            "symbol": symbol,
            "current": live_price or current,
            "lstm_prediction": lstm_pred,
            "prophet_prediction": prophet_pred,
            "final_prediction": final_pred,
            "signal": signal,
            "confidence": final_confidence,
            "sentiment_score": sentiment_score,
            "success_chance": classifier_prob,
            "timestamp": pd.Timestamp.now().isoformat()
        }

    except Exception as e:
        print(f"❌ run_prediction failed for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


if __name__ == "__main__":
    print("🚀 Starting Crypto Prediction API...")
    print(f"📊 Supporting {len(SYMBOL_MAP)} cryptocurrencies")
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)