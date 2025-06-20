import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from threading import Timer
from dotenv import load_dotenv
import csv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or "8021958251:AAEQMwfChSEZ3UXxsleI4A1dIquLhY20E0A"
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or "854634209"

PREDICTION_LOG_FILE = "prediction_log.csv"


def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        requests.post(url, data=payload)
    except Exception as e:
        print("❌ Telegram send error:", e)

def log_prediction(symbol, signal, price):
    file_exists = os.path.exists(PREDICTION_LOG_FILE)
    write_header = not file_exists or os.path.getsize(PREDICTION_LOG_FILE) == 0

    with open(PREDICTION_LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "symbol", "signal", "predicted_price"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            symbol.upper(),
            signal.capitalize(),
            round(price, 8)
        ])



def evaluate_accuracy(N=20, lookahead_minutes=60):
    try:
        df = pd.read_csv(PREDICTION_LOG_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', infer_datetime_format=True)
    except Exception as e:
        print("❌ Failed to read prediction log:", e)
        return

    results = []

    for _, row in df.tail(N).iterrows():
        symbol = row["symbol"]
        signal = row["signal"]
        predicted_price = float(row["predicted_price"])
        timestamp = row["timestamp"]

        future_time = timestamp + timedelta(minutes=lookahead_minutes)
        now = datetime.utcnow()

        if now < future_time:
            continue  # not enough time has passed

        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
            actual_price = float(requests.get(url).json()["price"])
        except:
            continue

        correct = (
            (signal.lower() == "buy" and actual_price > predicted_price) or
            (signal.lower() == "sell" and actual_price < predicted_price)
        )
        results.append(correct)

    if results:
        accuracy = 100 * sum(results) / len(results)
        message = f"📊 *Accuracy Report*Checked: {len(results)} predictions\nAccuracy: {accuracy:.2f}% over last {N} signals"
        print(message)
        send_telegram_message(message)
    else:
        print("⚠️ Not enough completed predictions to evaluate accuracy.")


def periodic_accuracy_check():
    evaluate_accuracy(N=20, lookahead_minutes=60)
    Timer(3600, periodic_accuracy_check).start()  # run every hour

def ensure_log_file():
    if not os.path.exists(PREDICTION_LOG_FILE):
        with open(PREDICTION_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "symbol", "signal", "predicted_price"])


if __name__ == "__main__":
    ensure_log_file()
    print("📈 Running hourly prediction accuracy monitor...")
    periodic_accuracy_check()
