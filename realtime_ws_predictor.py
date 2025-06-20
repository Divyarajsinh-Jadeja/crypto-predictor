import websocket
import json
import threading
from predictor_api import run_prediction
from gchat_bot import send_signal_change_alert  # ✅ Uses your existing bot

last_prices = {}
price_threshold = 0.01  # 1% change

def on_message(ws, message):
    data = json.loads(message)
    symbol = data['s']
    current_price = float(data['c'])

    # Get last price and compare
    last_price = last_prices.get(symbol)
    if last_price:
        price_change = abs(current_price - last_price) / last_price
        if price_change >= price_threshold:
            print(f"📉 {symbol} moved >1% ➜ {last_price:.2f} → {current_price:.2f}")

            try:
                result = run_prediction(symbol=symbol, live_price=current_price)
                if isinstance(result, dict) and "error" not in result:
                    send_signal_change_alert(result)
                    print(f"✅ Prediction done for {symbol}: {result.get('signal')}")
                else:
                    print(f"⚠️ Prediction error: {result}")
            except Exception as e:
                print(f"❌ Prediction failed for {symbol}: {e}")

    # Update last price
    last_prices[symbol] = current_price

def on_error(ws, error):
    print("❌ WebSocket error:", error)

def on_close(ws, code, msg):
    print("🔌 WebSocket closed:", code, msg)

def on_open(ws):
    payload = {
        "method": "SUBSCRIBE",
        "params": [
            "btcusdt@ticker",
            "ethusdt@ticker",
            "solusdt@ticker",
            "xrpusdt@ticker",
            "adausdt@ticker",
            "dogeusdt@ticker",
            "avaxusdt@ticker",
            "shibusdt@ticker",
            "dotusdt@ticker",
            "maticusdt@ticker",
            "trxusdt@ticker",
            "linkusdt@ticker",
            "ltcusdt@ticker",
            "uniusdt@ticker",
            "bchusdt@ticker",
            "xlmusdt@ticker",
            "nearusdt@ticker",
            "icpusdt@ticker",
            "aptusdt@ticker"
        ],
        "id": 1
    }
    ws.send(json.dumps(payload))

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

# Start in background
thread = threading.Thread(target=start_ws)
thread.start()
