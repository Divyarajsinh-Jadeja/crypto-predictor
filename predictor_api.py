import os
from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from train_model import add_features, compute_confidence, fetch_klines

app = Flask(__name__)

# Map coin names to Binance symbols
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


# 🔁 Predict for all coins
@app.route("/predict_all_lstm", methods=["GET"])
def predict_all_lstm():
    results = []

    for coin, symbol in SYMBOL_MAP.items():
        try:
            # 1. Fetch & prepare latest data
            df = fetch_klines(symbol)
            df = add_features(df)
            current = df["close"].iloc[-1]

            # 2. Load trained LSTM model + scaler
            model = load_model(f"models/{symbol}_lstm_model.h5")
            scaler = joblib.load(f"models/{symbol}scaler.pkl")

            # 3. Prepare input for LSTM
            X_pred = df[["close"]].values[-60:]
            X_pred_scaled = scaler.transform(X_pred)
            X_pred_scaled = np.reshape(X_pred_scaled, (1, X_pred_scaled.shape[0], 1))

            # 4. Predict next price
            pred_scaled = model.predict(X_pred_scaled)[0][0]
            predicted_price = scaler.inverse_transform([[pred_scaled]])[0][0]

            # 5. Load classification model for success chance
            try:
                clf = joblib.load(f"models/{symbol}_classifier.pkl")
                X_latest = df[["close", "volume", "rsi", "ema7", "momentum"]].iloc[-1:]
                proba = clf.predict_proba(X_latest)[0][1]  # probability of going up
                success_chance = float(round(proba * 100, 2))
            except Exception:
                success_chance = None

            # 6. Decision logic
            signal = "Buy" if predicted_price > current else "Sell"
            confidence = compute_confidence(current, predicted_price)

            # 7. Final result dictionary
            results.append({
                "coin": coin,
                "symbol": symbol,
                "current": float(current),
                "predicted_next": float(predicted_price),
                "signal": signal,
                "confidence": float(confidence),
                "success_chance": success_chance  # optional
            })

        except Exception as e:
            results.append({
                "coin": coin,
                "symbol": symbol,
                "error": f"{type(e).__name__}: {str(e)}"
            })

    return jsonify(results)


# 🔁 Predict for a single coin
@app.route("/predict", methods=["GET"])
def predict():
    coin = request.args.get("coin", "bitcoin")
    symbol = SYMBOL_MAP.get(coin)
    if not symbol:
        return jsonify({"error": "Unsupported coin"}), 400

    try:
        model = load_model(f"models/{symbol}_lstm_model.h5")
        scaler = joblib.load(f"models/{symbol}scaler.pkl")
        classifier = joblib.load(f"models/{symbol}_classifier.pkl")

        df = fetch_klines(symbol)
        df = add_features(df)

        last_60 = df[["close"]].values[-60:]
        scaled_input = scaler.transform(last_60)
        X_input = np.reshape(scaled_input, (1, 60, 1))

        pred_scaled = model.predict(X_input)[0][0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        current_price = df["close"].iloc[-1]

        features = df[["close", "volume", "rsi", "ema7", "momentum"]].iloc[-1].values.reshape(1, -1)
        prob_up = classifier.predict_proba(features)[0][1]
        confidence = compute_confidence(current_price, pred_price)
        signal = "Buy" if pred_price > current_price else "Sell"

        return jsonify({
            "coin": coin,
            "symbol": symbol,
            "current": float(current_price),
            "predicted_next": float(pred_price),
            "signal": signal,
            "confidence": float(confidence),
            "success_chance": float(round(prob_up * 100, 2))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
