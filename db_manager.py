import os
import pymongo
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "crypto_predictor"
COLLECTION_NAME = "predictions"

def get_db_connection():
    if not MONGO_URI:
        print("⚠️ MONGO_URI not found in environment variables. Using local fallback (if available) or failing.")
        return None
    try:
        client = pymongo.MongoClient(MONGO_URI)
        return client[DB_NAME]
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return None

def log_prediction_to_db(symbol, signal, price, confidence, success_chance):
    db = get_db_connection()
    if db is None:
        return
    
    record = {
        "timestamp": datetime.utcnow(),
        "symbol": symbol,
        "signal": signal,
        "predicted_price": float(price),
        "confidence": float(confidence),
        "success_chance": float(success_chance) if success_chance else 0.0
    }
    
    try:
        db[COLLECTION_NAME].insert_one(record)
        print(f"✅ Logged prediction for {symbol} to MongoDB")
    except Exception as e:
        print(f"❌ Failed to log to MongoDB: {e}")

def get_recent_predictions(limit=100):
    db = get_db_connection()
    if db is None:
        return []
    
    try:
        cursor = db[COLLECTION_NAME].find().sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"❌ Failed to fetch from MongoDB: {e}")
        return []
