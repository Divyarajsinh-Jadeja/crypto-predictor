# Crypto Predictor (Free AI Model)

## 🔮 Features
- Trains a crypto price prediction model using Binance data
- Predicts tomorrow's price using XGBoost
- Sends updates to Telegram
- Free hosting on Render

## 🚀 Quick Start

### 1. Train Model
```bash
python train_model.py
```

### 2. Run Prediction API
```bash
python predictor_api.py
```

### 3. Send Telegram Message
```bash
python telegram_bot.py
```

### 4. Deploy on Render
- Push this repo to GitHub
- Connect to [Render.com](https://render.com)
- Deploy with `render.yaml`

### 5. Schedule Hourly Predictions
Use a cron job (via worker, Linux cron, or GitHub Actions)

---

_This system is free, efficient, and perfect for solo use or Telegram bots._  
