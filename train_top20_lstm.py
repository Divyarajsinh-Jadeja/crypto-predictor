from train_model import fetch_klines, add_features, train_classifier
from train_model import train_lstm  # assuming this is your lstm training module
from train_model import train_prophet

coins = {
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

for coin, symbol in coins.items():
    try:
        print(f"📈 Training: {symbol}")
        df = fetch_klines(symbol)
        df = add_features(df)

        if df.empty or len(df) < 100:
            print(f"⚠️ Not enough data for {symbol}")
            continue

        train_lstm(df, symbol)
        train_classifier(df, symbol)
        print(f"✅ Trained both LSTM & classifier for {symbol}\n")

    except Exception as e:
        print(f"❌ Failed to train for {symbol}: {e}")
