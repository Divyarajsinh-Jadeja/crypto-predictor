# Indian Crypto Market Prediction System

## Overview
A comprehensive cryptocurrency prediction system adapted for the Indian market using WazirX INR trading pairs. The system combines LSTM neural networks, Prophet time series forecasting, and machine learning classifiers to predict price movements and generate trading signals.

## Key Features
- **Indian Market Focus**: Uses WazirX API for INR trading pairs
- **Multi-Model Approach**: LSTM + Prophet + Classifier ensemble
- **Real-time Monitoring**: WebSocket integration for live price tracking
- **Telegram Alerts**: Automated signal notifications
- **Accuracy Tracking**: Performance monitoring and evaluation
- **Top 20 Cryptocurrencies**: Covers major coins in Indian market

## Supported Cryptocurrencies (INR Pairs)
- Bitcoin (BTCINR)
- Ethereum (ETHINR)
- Binance Coin (BNBINR)
- Solana (SOLINR)
- Ripple (XRPINR)
- Cardano (ADAINR)
- Dogecoin (DOGEINR)
- Avalanche (AVXINR)
- Shiba Inu (SHIBINR)
- Polkadot (DOTINR)
- Polygon (MATICINR)
- Tron (TRXINR)
- Chainlink (LINKINR)
- Litecoin (LTCINR)
- Uniswap (UNIINR)
- Bitcoin Cash (BCHINR)
- Stellar (XLMINR)
- Near Protocol (NEARINR)
- Internet Computer (ICPINR)
- Aptos (APTINR)

## Installation & Setup

### 1. Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file:
```
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Model Training
```bash
python train_20_lstm.py
```
This will train LSTM models for all 20 supported cryptocurrencies using WazirX historical data.

### 4. Start the API Server
```bash
python predictor_api.py
```
API will be available at `http://localhost:5050`

### 5. Start Accuracy Monitoring
```bash
python accuracy_tracker.py
```
Runs hourly accuracy checks and sends reports via Telegram.

### 6. Real-time Monitoring (Optional)
```bash
python realtime_ws_predictor.py
```
Monitors live price changes and triggers predictions automatically.

## API Endpoints

### Get All Predictions
```
GET /predict_all_lstm
```
Returns predictions for all 20 supported cryptocurrencies.

### Get Single Prediction
```
GET /predict?coin=bitcoin
```
Returns prediction for a specific cryptocurrency.

### Health Check
```
GET /health
```
Checks if the API is running.

### Model Status
```
GET /models/status
```
Shows the status of all trained models.

## Architecture

### Data Flow
1. **Data Collection**: Fetches historical price data from WazirX API
2. **Feature Engineering**: Calculates RSI, EMA, momentum, and other technical indicators
3. **Model Training**: Trains LSTM, Prophet, and classifier models
4. **Prediction**: Combines multiple model outputs for final prediction
5. **Signal Generation**: Generates BUY/SELL signals based on predictions
6. **Monitoring**: Tracks prediction accuracy over time

### Model Ensemble
- **LSTM (60%)**: Primary neural network for price prediction
- **Prophet (40%)**: Time series forecasting for trend analysis
- **Classifier**: Provides success probability for signals

## Key Improvements for Indian Market

### 1. API Migration
- **From**: Binance USD pairs
- **To**: WazirX INR pairs
- **Benefit**: Direct INR trading without currency conversion

### 2. Market-Specific Features
- Adapted to Indian trading hours and patterns
- INR-denominated price predictions
- Local market volatility considerations

### 3. Enhanced Symbol Mapping
```python
COIN_ID_MAP_INR = {
    "btcinr": "bitcoin",
    "ethinr": "ethereum",
    # ... complete mapping for Indian pairs
}
```

## File Structure
```
├── models/                 # Trained model files
│   ├── btcinr_lstm_model.h5
│   ├── btcinr_classifier.pkl
│   └── btcinrscaler.pkl
├── predictor_api.py        # Main API server
├── train_20_lstm.py        # Model training script
├── accuracy_tracker.py     # Performance monitoring
├── realtime_ws_predictor.py # Live monitoring
├── requirements.txt        # Dependencies
└── prediction_log.csv      # Prediction history
```

## Usage Examples

### Manual Prediction
```bash
curl "http://localhost:5050/predict?coin=bitcoin"
```

### Batch Predictions
```bash
curl "http://localhost:5050/predict_all_lstm"
```

### Check Model Status
```bash
curl "http://localhost:5050/models/status"
```

## Monitoring & Alerts

### Telegram Integration
- Real-time trading signals
- Accuracy reports every hour
- Model performance summaries

### Accuracy Tracking
- Evaluates last 20 predictions
- 60-minute lookahead validation
- Automated performance reporting

## Best Practices

### 1. Model Retraining
- Retrain models weekly with fresh data
- Monitor accuracy degradation
- Update features based on market conditions

### 2. Risk Management
- Use predictions as one factor in trading decisions
- Implement position sizing based on confidence scores
- Set stop-loss orders for risk control

### 3. Data Quality
- Validate API responses before processing
- Handle missing data gracefully
- Monitor for data anomalies

## Troubleshooting

### Common Issues
1. **Model Files Missing**: Run `train_20_lstm.py` first
2. **API Timeouts**: Check WazirX API status
3. **Telegram Not Working**: Verify bot token and chat ID
4. **Low Accuracy**: Retrain models with more recent data

### Debug Mode
Set `debug=True` in Flask app for detailed error logging.

## Future Enhancements

### Planned Features
- Advanced technical indicators
- Sentiment analysis integration
- Portfolio optimization
- Mobile app development
- Paper trading simulation

### Market Expansion
- Add more Indian exchange integrations
- Support for futures and options
- Regional crypto regulations compliance

## Disclaimer
This system is for educational and research purposes. Cryptocurrency trading involves significant risk. Always conduct your own research and consider consulting with financial advisors before making investment decisions.