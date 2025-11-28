# Crypto Market Prediction System

## Overview
A comprehensive cryptocurrency prediction system that forecasts prices for major USDT trading pairs. The system combines **LSTM neural networks**, **Prophet time series forecasting**, and **XGBoost classifiers** to predict price movements and generate trading signals. It uses the **Binance API** for historical and real-time data and sends alerts via **Google Chat**.

## Key Features
- **Binance Integration**: Uses Binance API for USDT trading pairs.
- **Multi-Model Ensemble**: Combines LSTM (Price), Prophet (Trend), and XGBoost (Direction) for robust predictions.
- **Real-time Monitoring**: WebSocket integration for live price tracking and instant alerts.
- **Sentiment Analysis**: Incorporates news sentiment to adjust confidence scores.
- **Google Chat Alerts**: Automated signal notifications and daily reports.
- **Database Logging**: Stores predictions in MongoDB for historical analysis.

## Supported Cryptocurrencies (USDT Pairs)
The system supports the following 20 cryptocurrencies:
- Bitcoin (BTCUSDT)
- Ethereum (ETHUSDT)
- Binance Coin (BNBUSDT)
- Solana (SOLUSDT)
- Ripple (XRPUSDT)
- Cardano (ADAUSDT)
- Dogecoin (DOGEUSDT)
- Avalanche (AVAXUSDT)
- Shiba Inu (SHIBUSDT)
- Polkadot (DOTUSDT)
- Polygon (MATICUSDT)
- Tron (TRXUSDT)
- Chainlink (LINKUSDT)
- Litecoin (LTCUSDT)
- Uniswap (UNIUSDT)
- Bitcoin Cash (BCHUSDT)
- Stellar (XLMUSDT)
- Near Protocol (NEARUSDT)
- Internet Computer (ICPUSDT)
- Aptos (APTUSDT)

## Installation & Setup

### 1. Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory with the following variables:
```ini
# Google Chat Webhook for notifications
GOOGLE_CHAT_WEBHOOK_URL=your_google_chat_webhook_url

# MongoDB URI for storing predictions
MONGO_URI=your_mongodb_connection_string

# Optional: Port for the API (default is 5050)
PORT=5050
```

### 3. Model Training
Before running the API, you must train the models. This script fetches data from Binance and trains LSTM, Prophet, and Classifier models for all supported coins.
```bash
python train_model.py
```
*Note: This process may take some time as it fetches historical data and trains multiple models for each coin.*

### 4. Start the API Server
Start the Flask API to serve predictions:
```bash
python predictor_api.py
```
The API will be available at `http://localhost:5050`.

### 5. Real-time Monitoring (Optional)
To monitor live price changes and trigger predictions automatically based on market movement:
```bash
python realtime_ws_predictor.py
```

## API Endpoints

### Get All Predictions
```http
GET /predict_all_lstm
```
Returns predictions for all supported cryptocurrencies.

### Get Single Prediction
```http
GET /predict?coin=bitcoin
```
Returns prediction for a specific cryptocurrency.
**Parameters:**
- `coin`: The name of the coin (e.g., `bitcoin`, `ethereum`, `solana`).

### Send Report to Google Chat
```http
POST /send-to-chat
```
Triggers the bot to send a comprehensive prediction report to the configured Google Chat webhook.

### Health Check
```http
GET /health
```
Checks if the API is running.

### Model Status
```http
GET /models/status
```
Shows the status of trained models for each coin.

## Architecture

### Data Flow
1.  **Data Collection**: Fetches historical k-lines (candlestick data) from **Binance API**.
2.  **Feature Engineering**: Calculates RSI, EMA, MACD, Bollinger Bands, and other technical indicators.
3.  **Model Training**:
    *   **LSTM**: Predicts the exact future price based on sequence data.
    *   **Prophet**: Forecasts the general trend.
    *   **XGBoost Classifier**: Predicts the probability of the price going UP or DOWN.
4.  **Prediction**: The API aggregates these outputs.
    *   *Final Prediction* = 60% LSTM + 40% Prophet.
    *   *Confidence Score*: Adjusted based on the agreement between models and **Sentiment Analysis** (via Google News).
5.  **Notification**: Signals are sent to Google Chat.

## File Structure
```
├── models/                 # Directory containing trained model files (.h5, .pkl)
├── predictor_api.py        # Main Flask API server
├── train_model.py          # Script to fetch data and train all models
├── gchat_bot.py            # Logic for formatting and sending Google Chat messages
├── realtime_ws_predictor.py # WebSocket client for real-time Binance monitoring
├── sentiment_analyzer.py   # Module for fetching and analyzing crypto news sentiment
├── db_manager.py           # MongoDB interaction layer
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Troubleshooting

### Common Issues
1.  **"Missing files" in Model Status**: Ensure you have run `python train_model.py` successfully.
2.  **API Connection Failed**: Check if `predictor_api.py` is running and the port (default 5050) is not blocked.
3.  **Google Chat Errors**: Verify your `GOOGLE_CHAT_WEBHOOK_URL` in the `.env` file.
4.  **MongoDB Errors**: Ensure `MONGO_URI` is correct and your IP is whitelisted if using a cloud database.

## 🤖 Free Automation Setup

This project includes **GitHub Actions workflows** for complete automation:

- 🔄 **Auto-training**: Retrain models weekly with fresh data
- 📊 **Auto-predictions**: Send updates to Google Chat every 6 hours
- 💚 **Keep-alive**: Prevent your Render service from sleeping

**👉 See [AUTOMATION.md](AUTOMATION.md) for complete setup instructions**

Quick start:
1. Push code to GitHub
2. Add secrets: `GOOGLE_CHAT_WEBHOOK_URL` and `API_BASE_URL`
3. Enable GitHub Actions
4. Done! Everything runs automatically for **$0/month**

## Disclaimer
This system is for educational and research purposes only. Cryptocurrency trading involves significant risk. Always conduct your own research before making investment decisions.