import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from threading import Timer
from dotenv import load_dotenv

from gchat_bot import send_chat_message
from db_manager import get_recent_predictions

load_dotenv()

def evaluate_accuracy(N=20, lookahead_minutes=60):
    try:
        # Fetch from MongoDB
        predictions = get_recent_predictions(limit=N*2) # Fetch a bit more to ensure we have enough
        if not predictions:
            print("⚠️ No predictions found in database.")
            return

        df = pd.DataFrame(predictions)
        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as e:
        print("❌ Failed to read prediction log:", e)
        return

    results = []

    # Sort by timestamp ascending for processing
    df = df.sort_values("timestamp")

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
        message = f"📊 *Accuracy Report*\nChecked: {len(results)} predictions\nAccuracy: {accuracy:.2f}% over last {N} signals"
        print(message)
        send_chat_message(message)
    else:
        print("⚠️ Not enough completed predictions to evaluate accuracy.")


def periodic_accuracy_check():
    evaluate_accuracy(N=20, lookahead_minutes=60)
    Timer(3600, periodic_accuracy_check).start()  # run every hour


if __name__ == "__main__":
    print("📈 Running hourly prediction accuracy monitor (MongoDB)...")
    periodic_accuracy_check()
