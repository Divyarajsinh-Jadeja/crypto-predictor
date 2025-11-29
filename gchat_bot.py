import os
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_CHAT_WEBHOOK_URL = os.getenv("GOOGLE_CHAT_WEBHOOK_URL")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5050")

# Coin ID mapping for display names
COIN_ID_MAP = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT",
    "binancecoin": "BNBUSDT",
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

def get_usd_inr_rate():
    """Fetch current USD to INR exchange rate"""
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url, timeout=5)
        if response.ok:
            data = response.json()
            return data["rates"]["INR"]
    except Exception as e:
        print(f"⚠️ Failed to fetch USD/INR rate: {e}")
    return 89.39  # Default fallback rate

def format_number(value, decimals=2):
    """Format number with commas and decimals"""
    try:
        if value is None:
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except:
        return str(value)

def format_prediction(prediction, usd_to_inr):
    """Format a single prediction into a message section"""
    symbol = prediction.get("symbol", "UNKNOWN")
    
    # Clean symbol name (remove USDT)
    coin_name = symbol.replace("USDT", "")
    for name, code in COIN_ID_MAP.items():
        if code == symbol:
            coin_name = name.title()
            break
            
    current_price = prediction.get("current", 0)
    predicted_price = prediction.get("final_prediction", 0)
    signal = prediction.get("signal", "Hold")
    confidence = prediction.get("confidence", 0)
    success_chance = prediction.get("success_chance", 0)
    
    # Calculate prices in INR
    current_inr = current_price * usd_to_inr
    predicted_inr = predicted_price * usd_to_inr
    
    # Determine emojis based on signal
    if signal == "Buy":
        emoji = "🚀"
        signal_icon = "🟢"
    elif signal == "Sell":
        emoji = "📉"
        signal_icon = "�"
    else:
        emoji = "⚖️"
        signal_icon = "⚪"
        
    # Calculate percentage change
    if current_price > 0:
        change_pct = ((predicted_price - current_price) / current_price) * 100
    else:
        change_pct = 0
        
    change_icon = "📈" if change_pct >= 0 else "📉"
    
    # Format large INR numbers
    def format_inr(amount):
        if amount >= 10000000: # Crores
            return f"₹{amount/10000000:.2f}Cr"
        elif amount >= 100000: # Lakhs
            return f"₹{amount/100000:.2f}L"
        else:
            return f"₹{amount:,.0f}"

    msg = f"\n{emoji} *{coin_name} ({symbol})*\n"
    msg += f"💰 Current: ${format_number(current_price)} / {format_inr(current_inr)}\n"
    msg += f"🔮 Target: ${format_number(predicted_price)} / {format_inr(predicted_inr)}\n"
    msg += f"{signal_icon} Signal: *{signal}*\n"
    msg += f"{change_icon} Change: {change_pct:+.2f}%\n"
    msg += f"🎯 Confidence: {format_number(confidence)}%\n"
    msg += f"🟡 Success Chance: {format_number(success_chance)}%\n"
    
    return msg

def send_chat_message(text):
    """Send message to Google Chat webhook"""
    if not GOOGLE_CHAT_WEBHOOK_URL:
        print("❌ GOOGLE_CHAT_WEBHOOK_URL not found in environment variables")
        return
        
    try:
        headers = {"Content-Type": "application/json; charset=UTF-8"}
        data = {"text": text}
        response = requests.post(GOOGLE_CHAT_WEBHOOK_URL, json=data, headers=headers)
        if response.ok:
            print("✅ Message sent to Google Chat")
        else:
            print(f"❌ Failed to send message: {response.text}")
    except Exception as e:
        print(f"❌ Error sending message: {e}")

def send_all_predictions():
    """Fetch and send all crypto predictions via Google Chat"""
    try:
        # Check API health first - with extended timeout for Render wake-up
        health_url = f"{API_BASE_URL}/health"
        print(f"🔍 Checking API health at {health_url}...")
        
        # Try health check with retries (Render can take 30-60s to wake up)
        for attempt in range(3):
            try:
                print(f"  Attempt {attempt + 1}/3...")
                health_response = requests.get(health_url, timeout=90)  # 90 second timeout
                if health_response.ok:
                    print(f"✅ API is awake and healthy!")
                    break
                else:
                    print(f"⚠️ Health check returned status: {health_response.status_code}")
                    if attempt < 2:
                        print(f"  Waiting 30 seconds before retry...")
                        time.sleep(30)
            except requests.exceptions.Timeout:
                if attempt < 2:
                    print(f"⏳ Timeout on attempt {attempt + 1}. API might be waking up...")
                    print(f"  Waiting 45 seconds before retry...")
                    time.sleep(45)
                else:
                    send_chat_message(f"⚠️ API did not wake up after 3 attempts. Render might be experiencing issues.")
                    return
            except Exception as e:
                if attempt < 2:
                    print(f"⚠️ Health check error: {str(e)}")
                    print(f"  Waiting 30 seconds before retry...")
                    time.sleep(30)
                else:
                    send_chat_message(f"⚠️ API is not responding: {str(e)}\n\nTip: Check if your Render service is running.")
                    return

        # Fetch predictions in batches to prevent server timeout/OOM
        print("🔄 Fetching predictions in batches...")
        all_predictions = []
        
        # Split coins into batches of 3
        target_coins = [
            "bitcoin", "ethereum", "solana", "binancecoin", "ripple", 
            "cardano", "dogecoin", "avalanche", "shiba-inu", "polkadot",
            "polygon", "tron", "chainlink", "litecoin", "uniswap",
            "bitcoin-cash", "stellar", "near", "internet-computer", "aptos"
        ]
        
        batch_size = 3
        for i in range(0, len(target_coins), batch_size):
            batch = target_coins[i:i + batch_size]
            batch_str = ",".join(batch)
            print(f"  Fetching batch {i//batch_size + 1}: {batch_str}")
            
            try:
                # Use the batch endpoint
                batch_url = f"{API_BASE_URL}/predict_batch?coins={batch_str}"
                response = requests.get(batch_url, timeout=120) # 2 minute timeout per batch
                
                if response.ok:
                    batch_results = response.json()
                    all_predictions.extend(batch_results)
                    print(f"  ✅ Batch {i//batch_size + 1} success")
                else:
                    print(f"  ❌ Batch {i//batch_size + 1} failed: {response.status_code}")
                    # Add error placeholders for this batch
                    for coin in batch:
                        all_predictions.append({"symbol": coin.upper(), "error": f"Batch failed: {response.status_code}"})
                
                # Small delay to let server cool down
                time.sleep(2)
                
            except Exception as e:
                print(f"  ❌ Batch {i//batch_size + 1} error: {str(e)}")
                for coin in batch:
                    all_predictions.append({"symbol": coin.upper(), "error": f"Batch error: {str(e)}"})

        predictions = all_predictions

        # Get USD to INR rate
        usd_to_inr = get_usd_inr_rate()
        print(f"💱 Current USD/INR Rate: {usd_to_inr}")

        # Format message
        message = "📊 *Crypto Predictions Report*\n"
        message += f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"💱 USD/INR: {usd_to_inr}\n"
        
        success_count = sum(1 for p in predictions if "error" not in p)
        message += f"📈 Success: {success_count}/{len(predictions)} coins\n\n"

        failed_predictions = []

        for p in predictions:
            if "error" in p:
                failed_predictions.append(p)
                continue
            
            msg = format_prediction(p, usd_to_inr)
            if len(message) + len(msg) > 4000:  # Google Chat limit
                send_chat_message(message)
                message = ""
            message += msg

        if failed_predictions:
            message += "\n❌ *Failed Predictions:*\n"
            for p in failed_predictions:
                coin_name = p.get('symbol', 'Unknown')
                # Map symbol back to name if possible
                for name, symbol in COIN_ID_MAP.items():
                    if symbol == coin_name:
                        coin_name = name.replace("USDT", "")
                        break
                message += f"• {coin_name}: {p['error']}\n"

        message += "\n📱 Generated by Crypto Prediction Bot"
        send_chat_message(message)
        print("✅ Report sent to Google Chat")

    except Exception as e:
        print(f"❌ Error in send_all_predictions: {e}")
        send_chat_message(f"⚠️ Bot Error: {str(e)}")

if __name__ == "__main__":
    send_all_predictions()
