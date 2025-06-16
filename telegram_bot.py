import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = "8021958251:AAEQMwfChSEZ3UXxsleI4A1dIquLhY20E0A"
TELEGRAM_CHAT_ID = "854634209"

def get_usd_inr_rate():
    """
    Get USD to INR exchange rate from multiple free APIs
    """
    # Try multiple free APIs in order of preference
    apis = [
        {
            "url": "https://api.exchangerate-api.com/v4/latest/USD",
            "path": ["rates", "INR"]
        },
        {
            "url": "https://open.er-api.com/v6/latest/USD",
            "path": ["rates", "INR"]
        },
        {
            "url": "https://api.fxratesapi.com/latest?base=USD&symbols=INR",
            "path": ["rates", "INR"]
        }
    ]
    
    for api in apis:
        try:
            response = requests.get(api["url"], timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Navigate through the JSON path
            rate = data
            for key in api["path"]:
                rate = rate[key]
            
            if rate and isinstance(rate, (int, float)) and rate > 0:
                print(f"✅ Got USD-INR rate: {rate} from {api['url']}")
                return float(rate)
                
        except Exception as e:
            print(f"❌ Failed to get rate from {api['url']}: {str(e)}")
            continue
    
    # Fallback rate if all APIs fail
    print("⚠️ Using fallback USD-INR rate: 83.0")
    return 83.0

def escape_md(text):
    """
    Escapes all special characters required by MarkdownV2.
    """
    if not isinstance(text, str):
        text = str(text)

    # Updated to include period (.) in characters to escape
    to_escape = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(to_escape)}])', r'\\\1', text)

def format_number(number, decimals=4):
    """
    Format number and escape for MarkdownV2
    Automatically adjusts decimal places for very small numbers
    """
    if number == 0:
        formatted = "0.0000"
    elif number < 0.0001:
        # For very small numbers, use up to 8 decimal places
        formatted = f"{number:.8f}".rstrip('0').rstrip('.')
        if '.' not in formatted:
            formatted += ".0"
    elif number < 1:
        # For numbers less than 1, use 6 decimal places
        formatted = f"{number:.6f}".rstrip('0').rstrip('.')
        if '.' not in formatted:
            formatted += ".0"
    else:
        # For larger numbers, use the specified decimal places
        formatted = f"{number:.{decimals}f}"
    
    return escape_md(formatted)

def format_currency_inr(amount):
    """
    Format INR amount with commas and escape for MarkdownV2
    Automatically adjusts decimal places for very small amounts
    """
    if amount == 0:
        formatted = "0.00"
    elif amount < 0.01:
        # For very small amounts, use up to 6 decimal places
        formatted = f"{amount:.6f}".rstrip('0').rstrip('.')
        if '.' not in formatted:
            formatted += ".0"
    elif amount < 1:
        # For amounts less than 1, use 4 decimal places
        formatted = f"{amount:.4f}".rstrip('0').rstrip('.')
        if '.' not in formatted:
            formatted += ".0"
    else:
        # For larger amounts, use standard 2 decimal places with commas
        formatted = f"{amount:,.2f}"
    
    return escape_md(formatted)

def format_prediction(coin_data, usd_to_inr):
    try:
        coin = escape_md(coin_data["coin"].capitalize())
        symbol = escape_md(coin_data["symbol"])
        current = float(coin_data["current"])
        predicted = float(coin_data["predicted_next"])
        signal = escape_md(coin_data["signal"])
        confidence = float(coin_data["confidence"])
        success_chance = coin_data.get("success_chance", None)

        current_inr = current * usd_to_inr
        predicted_inr = predicted * usd_to_inr

        message = f"*{coin}* \\({symbol}\\)\n"
        message += f"💰 Current: \\${format_number(current)} / ₹{format_currency_inr(current_inr)}\n"
        message += f"🔮 Predicted: \\${format_number(predicted)} / ₹{format_currency_inr(predicted_inr)}\n"
        message += f"📈 Signal: {signal}\n"
        message += f"🎯 Confidence: {format_number(confidence, 2)}%\n"

        if success_chance is not None:
            message += f"📊 Success Chance: {format_number(success_chance, 2)}%\n"

        return message
    except Exception as e:
        error_msg = escape_md(str(e))
        coin_name = escape_md(coin_data.get('coin', 'Unknown'))
        return f"⚠️ Error formatting {coin_name}: {error_msg}"

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID env vars")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "MarkdownV2"
    }

    try:
        response = requests.post(url, json=payload)
        if not response.ok:
            print(f"❌ Telegram error: {response.text}")
        else:
            print("✅ Message sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {str(e)}")

def send_all_predictions():
    try:
        res = requests.get("http://127.0.0.1:5050/predict_all_lstm")
        predictions = res.json()
        usd_to_inr = get_usd_inr_rate()

        if not predictions:
            send_telegram_message(escape_md("⚠️ No predictions available."))
            return

        message = "📊 *LSTM Crypto Predictions \\(Top Coins\\)*\n\n"
        for coin_data in predictions:
            if "error" in coin_data:
                coin = escape_md(coin_data['coin'].capitalize())
                error = escape_md(coin_data["error"])
                message += f"❌ *{coin}* \\- `{error}`\n\n"
            else:
                section = format_prediction(coin_data, usd_to_inr)
                if coin_data.get("success_chance", 0) > 90:
                    section = f"🟩 *High Confidence*\n{section}"
                message += section + "\n"

        send_telegram_message(message)

    except Exception as e:
        error_msg = escape_md(f"🚨 Prediction fetch failed: {str(e)}")
        send_telegram_message(error_msg)

if __name__ == "__main__":
    send_all_predictions()