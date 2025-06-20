import os
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or "8021958251:AAEQMwfChSEZ3UXxsleI4A1dIquLhY20E0A"
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or "854634209"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5050")

last_known_signals = {}

COIN_ID_MAP = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin",
    "XRPUSDT": "ripple",
    "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin",
    "AVAXUSDT": "avalanche",
    "SHIBUSDT": "shiba-inu",
    "DOTUSDT": "polkadot",
    "MATICUSDT": "polygon",
    "TRXUSDT": "tron",
    "LINKUSDT": "chainlink",
    "LTCUSDT": "litecoin",
    "UNIUSDT": "uniswap",
    "BCHUSDT": "bitcoin-cash",
    "XLMUSDT": "stellar",
    "NEARUSDT": "near",
    "ICPUSDT": "internet-computer",
    "APTUSDT": "aptos"
}


def send_signal_change_alert(coin_data):
    """Send alert if signal has changed"""
    try:
        symbol = coin_data["symbol"].lower()
        current_signal = coin_data["signal"].lower()
        previous_signal = last_known_signals.get(symbol)

        if current_signal != previous_signal:
            usd_to_inr = get_usd_inr_rate()
            message = format_prediction(coin_data, usd_to_inr)
            header = "🔔 *Signal Changed*\n\n" if previous_signal is not None else "📡 *New Signal Detected*\n\n"
            send_chat_message(header + message)
            last_known_signals[symbol] = current_signal

    except Exception as e:
        print(f"❌ Failed to send signal change alert for {coin_data.get('symbol')}: {e}")


def get_usd_inr_rate():
    """Get USD to INR exchange rate from multiple APIs with fallback"""
    apis = [
        {"url": "https://api.exchangerate-api.com/v4/latest/USD", "path": ["rates", "INR"]},
        {"url": "https://open.er-api.com/v6/latest/USD", "path": ["rates", "INR"]},
        {"url": "https://api.fxratesapi.com/latest?base=USD&symbols=INR", "path": ["rates", "INR"]},
        {"url": "https://api.currencyapi.com/v3/latest?apikey=YOUR_API_KEY&currencies=INR&base_currency=USD",
         "path": ["data", "INR", "value"]}
    ]

    for api in apis:
        try:
            response = requests.get(api["url"], timeout=10)
            response.raise_for_status()
            data = response.json()

            rate = data
            for key in api["path"]:
                rate = rate[key]

            if rate and isinstance(rate, (int, float)) and rate > 70 and rate < 100:  # Sanity check
                print(f"✅ Got USD/INR rate: {rate}")
                return float(rate)

        except Exception as e:
            print(f"⚠️ Failed to get rate from {api['url']}: {e}")
            continue

    print("⚠️ All exchange rate APIs failed, using fallback rate")
    return 83.50  # Updated fallback rate


# def escape_md(text):
#     """Escape special characters for Telegram MarkdownV2"""
#     if text is None:
#         return "N/A"
#
#     # Characters that need escaping in MarkdownV2
#     to_escape = r'_*[]()~`>#+-=|{}.!'
#     escaped = re.sub(f'([{re.escape(to_escape)}])', r'\\\1', str(text))
#     return escaped


def format_number(number, decimals=4):
    """Format numbers with appropriate precision and escape for MarkdownV2"""
    if number is None:
        return "N/A"

    try:
        number = float(number)
        if number == 0:
            formatted = "0.0000"
        elif abs(number) < 0.0001:
            formatted = f"{number:.8f}".rstrip('0').rstrip('.')
        elif abs(number) < 1:
            formatted = f"{number:.6f}".rstrip('0').rstrip('.')
        elif abs(number) < 1000:
            formatted = f"{number:.{decimals}f}"
        else:
            formatted = f"{number:,.{decimals}f}"
        return formatted
    except (ValueError, TypeError):
        return "N/A"


def format_currency_inr(amount):
    """Format INR currency amounts and escape for MarkdownV2"""
    if amount is None:
        return "N/A"

    try:
        amount = float(amount)
        if amount == 0:
            formatted = "0.00"
        elif amount < 0.01:
            formatted = f"{amount:.6f}".rstrip('0').rstrip('.')
        elif amount < 1:
            formatted = f"{amount:.4f}".rstrip('0').rstrip('.')
        elif amount < 100000:  # Less than 1 lakh
            formatted = f"{amount:,.2f}"
        else:  # Format in lakhs/crores for large amounts
            if amount >= 10000000:  # 1 crore
                crores = amount / 10000000
                formatted = f"{crores:.2f}Cr"
            else:  # lakhs
                lakhs = amount / 100000
                formatted = f"{lakhs:.2f}L"
        return formatted
    except (ValueError, TypeError):
        return "N/A"


def get_trend_emoji(current, predicted):
    """Get trend emoji based on price movement"""
    try:
        change_percent = ((predicted - current) / current) * 100
        if change_percent > 5:
            return "🚀"
        elif change_percent > 2:
            return "📈"
        elif change_percent > 0:
            return "⬆️"
        elif change_percent > -2:
            return "⬇️"
        elif change_percent > -5:
            return "📉"
        else:
            return "💥"
    except:
        return "➡️"


def format_prediction(coin_data, usd_to_inr):
    """Format individual coin prediction for Telegram"""
    try:
        symbol = coin_data.get("symbol", "").upper()
        coin_name = COIN_ID_MAP.get(symbol, symbol)  # fallback to symbol if no mapping
        coin = coin_name.replace("-", " ").title()
        current = float(coin_data["current"])
        predicted = float(coin_data.get("predicted_next", coin_data.get("final_prediction", 0)))
        signal = coin_data["signal"]
        confidence = float(coin_data.get("confidence", 0))
        success_chance = coin_data.get("success_chance")
        prophet_pred = coin_data.get("prophet_prediction")
        final_pred = coin_data.get("final_prediction")

        # Calculate INR values
        current_inr = current * usd_to_inr
        predicted_inr = predicted * usd_to_inr

        # Calculate change percentage
        change_percent = ((predicted - current) / current) * 100
        trend_emoji = get_trend_emoji(current, predicted)

        # Build message - using escape_md for the symbol part
        message = f"{trend_emoji} *{coin}* {'('}{symbol}{')'}\n"
        message += f"💰 Current: ${format_number(current)} / ₹{format_currency_inr(current_inr)}\n"

        if final_pred and final_pred != predicted:
            final_inr = final_pred * usd_to_inr
            message += f"🎯 Final: ${format_number(final_pred)} / ₹{format_currency_inr(final_inr)}\n"
        else:
            message += f"🔮 LSTM: ${format_number(predicted)} / ₹{format_currency_inr(predicted_inr)}\n"

        if prophet_pred and prophet_pred != predicted:
            prophet_inr = prophet_pred * usd_to_inr
            message += f"📐 Prophet: ${format_number(prophet_pred)} / ₹{format_currency_inr(prophet_inr)}\n"

        # Signal with emoji
        signal_emoji = "🟢" if signal.lower() == "buy" else "🔴"
        message += f"{signal_emoji} Signal: *{signal}*\n"

        # Change percentage
        change_emoji = "📈" if change_percent > 0 else "📉"
        message += f"{change_emoji} Change: {format_number(change_percent, 2)}%\n"

        message += f"🎯 Confidence: {format_number(confidence, 2)}%\n"

        if success_chance is not None:
            success_emoji = "🟢" if success_chance > 70 else "🟡" if success_chance > 50 else "🔴"
            message += f"{success_emoji} Success: {format_number(success_chance, 2)}%\n"

        return message

    except Exception as e:
        error_msg = str(e)
        coin_name = coin_data.get('symbol') or coin_data.get('coin') or 'Unknown'
        return f"⚠️ Error formatting {coin_name}: {error_msg}"


GOOGLE_CHAT_WEBHOOK_URL = os.getenv(
    "GOOGLE_CHAT_WEBHOOK_URL") or "https://chat.googleapis.com/v1/spaces/AAQAeDW1n5I/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=bx4d3xHJi_rB2BWHQRdD_vWOuRexklVvOUL7Ywd0108"


def send_chat_message(message):
    """Send message to Google Chat via webhook"""
    if not GOOGLE_CHAT_WEBHOOK_URL.startswith("https://"):
        print("❌ Invalid Google Chat Webhook URL")
        return False

    payload = {"text": message}

    try:
        response = requests.post(GOOGLE_CHAT_WEBHOOK_URL, json=payload, timeout=10)
        if response.ok:
            print("✅ Message sent to Google Chat successfully!")
            return True
        else:
            print(f"❌ Google Chat error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to send message to Google Chat: {str(e)}")
        return False


def send_all_predictions():
    """Fetch and send all crypto predictions via Telegram"""
    try:
        # Check API health first
        health_url = f"{API_BASE_URL}/health"
        try:
            health_response = requests.get(health_url, timeout=10)
            if not health_response.ok:
                raise Exception(f"API health check failed: {health_response.status_code}")
        except Exception as e:
            send_chat_message(f"⚠️ API is not responding: {str(e)}")
            return

        # Fetch predictions
        predictions_url = f"{API_BASE_URL}/predict_all_lstm"
        print(f"📡 Fetching predictions from {predictions_url}")

        response = requests.get(predictions_url, timeout=120)  # Increased timeout
        response.raise_for_status()

        data = response.json()
        predictions = data.get("predictions", data) if isinstance(data, dict) else data

        if not predictions:
            send_chat_message("⚠️ No predictions available.")
            return

        # Get exchange rate
        usd_to_inr = get_usd_inr_rate()

        # Build header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_coins = len(predictions)
        successful = len([p for p in predictions if "error" not in p])

        message = f"📊 *Crypto Predictions Report*\n"
        message += f"🕐 Time: {timestamp}\n"
        message += f"💱 USD/INR: {format_number(usd_to_inr, 2)}\n"
        message += f"📈 Success: {successful}/{total_coins} coins\n\n"

        # Separate successful predictions from errors
        successful_predictions = []
        error_predictions = []

        for coin_data in predictions:
            if "error" in coin_data:
                error_predictions.append(coin_data)
            elif "symbol" not in coin_data or "signal" not in coin_data or "current" not in coin_data:
                coin_data["error"] = "Missing required fields"
                error_predictions.append(coin_data)
            else:
                successful_predictions.append(coin_data)
        # Sort successful predictions by success chance (if available)
        successful_predictions.sort(
            key=lambda x: x.get("success_chance", 0),
            reverse=True
        )

        # Add successful predictions
        for coin_data in successful_predictions:
            section = format_prediction(coin_data, usd_to_inr)

            # Highlight high confidence predictions
            success_chance = coin_data.get("success_chance", 0)
            if success_chance and success_chance > 85:
                section = f"🌟 *HIGH CONFIDENCE*\n{section}"
            elif success_chance and success_chance > 75:
                section = f"⭐ *Good Confidence*\n{section}"

            message += section + "\n"

        # Add errors at the end
        if error_predictions:
            message += "❌ *Failed Predictions:*\n"
            for coin_data in error_predictions:
                coin = coin_data['coin'].replace("-", " ").title()
                error = coin_data["error"][:100]  # Truncate long errors
                message += f"• {coin}: {error}\n"

        # Add footer
        message += f"\n📱 Generated by Crypto Prediction Bot"

        send_chat_message(message)

    except requests.exceptions.RequestException as e:
        error_msg = f"🚨 Network error fetching predictions: {str(e)}"
        send_chat_message(error_msg)
        print(f"❌ {error_msg}")
    except Exception as e:
        error_msg = f"🚨 Unexpected error: {str(e)}"
        send_chat_message(error_msg)
        print(f"❌ {error_msg}")


def test_telegram_connection():
    """Test Telegram bot connection"""
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your_default_token":
        print("❌ TELEGRAM_TOKEN not set")
        return False

    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "your_default_chat_id":
        print("❌ TELEGRAM_CHAT_ID not set")
        return False

    try:
        # Test bot info
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
        response = requests.get(url, timeout=10)

        if response.ok:
            bot_info = response.json()
            print(f"✅ Bot connected: {bot_info['result']['first_name']}")

            # Send test message
            test_message = f"🤖 *Bot Test Message*\n\nConnection successful{'!'}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            return send_chat_message(test_message)
        else:
            print(f"❌ Bot connection failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Telegram test failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            print("🧪 Testing Telegram connection...")
            test_telegram_connection()
        elif sys.argv[1] == "signal":
            print("📡 Fetching predictions for signal change alerts...")
            try:
                response = requests.get(f"{API_BASE_URL}/predict_all_lstm", timeout=60)
                response.raise_for_status()
                predictions = response.json().get("predictions", [])
                for coin in predictions:
                    if "error" not in coin:
                        send_signal_change_alert(coin)
            except Exception as e:
                send_chat_message(f"❌ Error fetching signal predictions: {str(e)}")
    else:
        print("📤 Sending crypto predictions...")
        send_all_predictions()
