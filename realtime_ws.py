import websocket
import json
import threading
from predictor_api import run_prediction  # your function that returns prediction & confidence

def on_message(ws, message):
    data = json.loads(message)
    symbol = data['s']
    price = float(data['c'])  # current price

    print(f"[{symbol}] Real-time Price: {price}")

    # You can now call your predictor here
    prediction = run_prediction(symbol, live_price=price)  # Pass price if needed
    print("🔮", prediction)

    # Optionally send to Telegram

def on_error(ws, error):
    print("❌ Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket closed")

def on_open(ws):
    payload = {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@ticker"],  # You can add more like 'ethusdt@ticker'
        "id": 1
    }
    ws.send(json.dumps(payload))

def start_ws():
    socket_url = "wss://stream.binance.com:9443/ws"
    ws = websocket.WebSocketApp(socket_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open)
    ws.run_forever()

# Run in thread to keep it alive
thread = threading.Thread(target=start_ws)
thread.start()
