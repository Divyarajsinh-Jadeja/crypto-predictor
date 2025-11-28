# How the Crypto Prediction System Works

## 📚 Table of Contents
1. [System Overview](#system-overview)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [Automation Workflow](#automation-workflow)
5. [Example Scenario](#example-scenario)

---

## System Overview

### The Big Picture

```
┌─────────────────┐
│  Binance API    │  ← Fetches real-time crypto prices
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              YOUR RENDER SERVER                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  predictor_api.py (Flask API)                     │  │
│  │  - Receives requests                              │  │
│  │  - Loads trained AI models                        │  │
│  │  - Makes predictions                              │  │
│  │  - Returns JSON results                           │  │
│  └──────────────┬───────────────────────────────────┘  │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Trained Models (in models/ folder)              │  │
│  │  - LSTM: Predicts exact future price             │  │
│  │  - Prophet: Predicts trend direction             │  │
│  │  - XGBoost: Predicts up/down probability         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  GitHub Actions (Automation)        │
│  ┌─────────────────────────────┐   │
│  │ Every 6 hours:              │   │
│  │ 1. Call Render API          │   │
│  │ 2. Get predictions          │   │
│  │ 3. Format message           │   │
│  │ 4. Send to Google Chat      │   │
│  └─────────────┬───────────────┘   │
└────────────────┼───────────────────┘
                 │
                 ▼
      ┌──────────────────┐
      │  Google Chat     │  ← You receive predictions
      └──────────────────┘
```

---

## Component Details

### 1. **Binance API** (External Data Source)
**What it does**: Provides historical cryptocurrency price data

**Example Request**:
```
GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d
```

**Response**: Array of price data like:
```json
[
  [
    1701388800000,  // Timestamp
    "42000.50",     // Open price
    "43000.75",     // High
    "41500.25",     // Low
    "42500.00",     // Close
    "1250.5"        // Volume
  ],
  ...
]
```

---

### 2. **train_model.py** (Model Training Script)

**When it runs**:
- Manually: `python train_model.py`
- Automatically: Every Sunday at 2 AM UTC (via GitHub Actions)

**What it does**:

```python
# Step 1: Fetch 5 years of historical data for BTCUSDT
df = fetch_klines("BTCUSDT")
# Result: DataFrame with ~1,825 rows (5 years * 365 days)

# Step 2: Add technical indicators
df = add_features(df)
# Adds: RSI, EMA, MACD, Bollinger Bands, etc.

# Step 3: Train 3 models per coin
train_lstm(df, "BTCUSDT")        # Saves: models/BTCUSDT_lstm_model.h5
train_prophet(df, "BTCUSDT")     # Saves: models/BTCUSDT_prophet.pkl
train_classifier(df, "BTCUSDT")  # Saves: models/BTCUSDT_classifier.pkl
```

**Output**: 80 files in `models/` folder (20 coins × 4 files each)

---

### 3. **predictor_api.py** (Flask Web Server)

**Where it runs**: On Render.com at `https://crypto-predictor-akpl.onrender.com`

**Endpoints**:

#### `/health` - Check if server is running
```bash
curl https://crypto-predictor-akpl.onrender.com/health
```
Response:
```json
{"status": "ok", "message": "API is live"}
```

#### `/predict?coin=bitcoin` - Get prediction for one coin
```bash
curl https://crypto-predictor-akpl.onrender.com/predict?coin=bitcoin
```
Response:
```json
{
  "symbol": "BTCUSDT",
  "current": 91334.66,
  "lstm_prediction": 92500.57,
  "prophet_prediction": 116744.36,
  "final_prediction": 102198.08,
  "signal": "Buy",
  "confidence": 11.89,
  "success_chance": 26.04,
  "sentiment_score": 0.0,
  "timestamp": "2025-11-28T12:01:24"
}
```

#### `/predict_all_lstm` - Get predictions for all 20 coins
Returns array of 20 predictions (one for each coin).

---

### 4. **gchat_bot.py** (Notification Script)

**When it runs**:
- Manually: `python gchat_bot.py`
- Automatically: Every 6 hours via GitHub Actions

**What it does**:

```python
# Step 1: Call your API
response = requests.get(f"{API_BASE_URL}/predict_all_lstm")
predictions = response.json()

# Step 2: Get USD to INR exchange rate
usd_to_inr = get_usd_inr_rate()  # Returns ~89.39

# Step 3: Format each prediction
for coin in predictions:
    message += format_prediction(coin, usd_to_inr)
    # Converts USD prices to INR
    # Adds emojis and formatting

# Step 4: Send to Google Chat
requests.post(GOOGLE_CHAT_WEBHOOK_URL, json={"text": message})
```

---

## Data Flow

### Complete Prediction Flow (Step-by-Step)

```
USER ACTION: GitHub Actions triggers at 6 AM UTC
    │
    ▼
1. GitHub runs gchat_bot.py
    │
    ▼
2. gchat_bot.py calls: https://crypto-predictor-akpl.onrender.com/predict_all_lstm
    │
    ▼
3. predictor_api.py receives request
    │
    ▼
4. For EACH of 20 coins (e.g., Bitcoin):
    │
    ├─▶ 4a. Fetch latest 60 days of data from Binance
    │       Result: Array of [timestamp, open, high, low, close, volume]
    │
    ├─▶ 4b. Calculate technical indicators (RSI, EMA, etc.)
    │       Result: DataFrame with 20+ features
    │
    ├─▶ 4c. Load LSTM model from models/BTCUSDT_lstm_model.h5
    │       Predict: Next day's price (e.g., $92,500)
    │
    ├─▶ 4d. Load Prophet model from models/BTCUSDT_prophet.pkl
    │       Predict: Trend (e.g., $116,744)
    │
    ├─▶ 4e. Load XGBoost classifier from models/BTCUSDT_classifier.pkl
    │       Predict: Probability of going UP (e.g., 26%)
    │
    ├─▶ 4f. Fetch news sentiment for "Bitcoin"
    │       Google News → TextBlob analysis → Score: -0.1 to +0.1
    │
    ├─▶ 4g. Combine predictions:
    │       Final = (60% × LSTM) + (40% × Prophet)
    │       Final = (0.6 × 92500) + (0.4 × 116744) = $102,198
    │
    ├─▶ 4h. Calculate confidence:
    │       Confidence = |Final - Current| / Current × 100
    │       Confidence = |102198 - 91334| / 91334 × 100 = 11.89%
    │
    └─▶ 4i. Adjust with sentiment:
          If sentiment > 0: Boost confidence by 5-10%
          If sentiment < 0: Reduce confidence
    │
    ▼
5. predictor_api.py returns JSON array of 20 predictions
    │
    ▼
6. gchat_bot.py receives all predictions
    │
    ▼
7. For EACH prediction:
    │
    ├─▶ Convert USD to INR (multiply by 89.39)
    ├─▶ Add emojis (🚀 for big gains, 📉 for drops)
    ├─▶ Format numbers with commas
    └─▶ Create message section
    │
    ▼
8. Final message looks like:
    ┌─────────────────────────────────────┐
    │ 📊 Crypto Predictions Report        │
    │ 🕐 Time: 2025-11-28 06:00:00       │
    │ 💱 USD/INR: 89.39                  │
    │                                     │
    │ 🚀 Bitcoin (BTCUSDT)               │
    │ 💰 Current: $91,334 / ₹81.6L      │
    │ 🔮 LSTM: $92,500 / ₹82.7L          │
    │ 🟢 Signal: Buy                     │
    │ 📈 Change: +11.89%                 │
    │ 🎯 Confidence: 65.5%               │
    │ 🟡 Success: 26%                    │
    │                                     │
    │ [... 19 more coins ...]            │
    └─────────────────────────────────────┘
    │
    ▼
9. Send to Google Chat webhook
    │
    ▼
10. YOU receive the message on your phone/computer! 📱
```

---

## Automation Workflow

### How GitHub Actions Works

#### Workflow 1: Send Predictions (Every 6 hours)

```yaml
on:
  schedule:
    - cron: '0 0,6,12,18 * * *'  # 12 AM, 6 AM, 12 PM, 6 PM UTC
```

**What happens**:

```
12:00 AM UTC (5:30 AM IST)
    ↓
GitHub's servers create a virtual machine (Ubuntu)
    ↓
Install Python 3.10
    ↓
Install requests and python-dotenv
    ↓
Run: python gchat_bot.py
    ↓
Environment variables injected:
  - GOOGLE_CHAT_WEBHOOK_URL = (from GitHub Secrets)
  - API_BASE_URL = https://crypto-predictor-akpl.onrender.com
    ↓
gchat_bot.py executes (calls API, formats, sends)
    ↓
Google Chat receives message
    ↓
GitHub machine shuts down
```

**Total time**: ~30 seconds  
**Cost**: FREE (uses ~2 minutes of 2000 free minutes/month)

---

#### Workflow 2: Train Models (Every Sunday)

```yaml
on:
  schedule:
    - cron: '0 2 * * 0'  # Every Sunday at 2 AM UTC
```

**What happens**:

```
Sunday 2:00 AM UTC (7:30 AM IST)
    ↓
GitHub creates Ubuntu virtual machine
    ↓
Install Python + TensorFlow + Prophet + XGBoost
    ↓
Run: python train_model.py
    ↓
For each of 20 coins:
  - Fetch 5 years of Binance data
  - Train LSTM (15 min)
  - Train Prophet (5 min)
  - Train XGBoost (2 min)
  Total: ~22 minutes per coin × 20 = ~7 hours
    ↓
Save 80 model files to models/ folder
    ↓
Commit and push to GitHub
    ↓
Models are now updated in your repository
    ↓
Render auto-deploys with new models
```

**Total time**: ~2 hours (with parallel processing)  
**Cost**: FREE (uses ~120 minutes of 2000/month)

---

#### Workflow 3: Keep API Alive (Every 10 minutes)

```yaml
on:
  schedule:
    - cron: '*/10 * * * *'  # Every 10 minutes
```

**What happens**:

```
Every 10 minutes
    ↓
GitHub: curl https://crypto-predictor-akpl.onrender.com/health
    ↓
Render receives request
    ↓
If sleeping: Wake up (takes 30 seconds)
If awake: Respond immediately
    ↓
Render stays awake for 15 more minutes
```

**Result**: Your API never sleeps!  
**Cost**: FREE (uses ~4 minutes/day = 120 min/month)

---

## Example Scenario

### Complete 24-Hour Cycle

```
🕐 12:00 AM UTC (5:30 AM IST)
   ├─ Keep-Alive pings Render ✓
   └─ Send Predictions runs ✓
       → You receive predictions in Google Chat

🕑 12:10 AM - Keep-Alive ping
🕒 12:20 AM - Keep-Alive ping
🕓 12:30 AM - Keep-Alive ping
...

🕕 6:00 AM UTC (11:30 AM IST)
   ├─ Keep-Alive pings Render ✓
   └─ Send Predictions runs ✓
       → You receive predictions in Google Chat

🕛 12:00 PM UTC (5:30 PM IST)
   ├─ Keep-Alive pings Render ✓
   └─ Send Predictions runs ✓
       → You receive predictions in Google Chat

🕕 6:00 PM UTC (11:30 PM IST)
   ├─ Keep-Alive pings Render ✓
   └─ Send Predictions runs ✓
       → You receive predictions in Google Chat

📅 Sunday 2:00 AM UTC (7:30 AM IST)
   └─ Train Models runs ✓
       → All models retrained with latest data
       → New models pushed to GitHub
       → Render auto-deploys
```

---

## Key Concepts

### 1. **Why 3 Models?**

Each model is good at different things:

- **LSTM**: Best for exact price prediction based on patterns
- **Prophet**: Best for understanding long-term trends
- **XGBoost**: Best for binary decisions (up/down)

**Combined**: More accurate than any single model!

### 2. **Why Sentiment Analysis?**

News affects crypto prices:
- Good news → Prices rise
- Bad news → Prices fall

We adjust confidence based on news sentiment to avoid false signals.

### 3. **Why MongoDB? (Optional)**

Stores prediction history so you can:
- Track accuracy over time
- Analyze which signals were correct
- Improve the models

**Note**: Works fine without MongoDB too!

### 4. **Why GitHub Actions?**

- ✅ Completely FREE
- ✅ No server needed
- ✅ Runs on GitHub's infrastructure
- ✅ Automatic scheduling
- ✅ Easy to configure

### 5. **Why Render?**

- ✅ Free hosting tier
- ✅ Auto-deploys from GitHub
- ✅ Provides HTTPS URL
- ✅ No credit card required

---

## Summary

**Your system is a fully automated crypto prediction machine that**:

1. 🤖 Trains AI models weekly with fresh data
2. 📊 Makes predictions 4 times daily
3. 📱 Sends formatted reports to Google Chat
4. 💚 Stays online 24/7
5. 💰 Costs $0/month

**All running on**:
- GitHub Actions (automation)
- Render (API hosting)
- Binance (data source)
- Google Chat (notifications)

**Without you doing anything!** 🚀

---

## Quick Reference

| Component | Purpose | Runs On | Frequency |
|-----------|---------|---------|-----------|
| `train_model.py` | Train AI models | GitHub Actions | Weekly |
| `predictor_api.py` | Serve predictions | Render | Always |
| `gchat_bot.py` | Send notifications | GitHub Actions | Every 6h |
| Keep-alive | Prevent sleep | GitHub Actions | Every 10min |

**Everything is automated. You just receive predictions!** ✨
