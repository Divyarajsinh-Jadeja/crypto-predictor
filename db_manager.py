import os
import pymongo
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "crypto_predictor"
COLLECTION_NAME = "predictions"
HISTORICAL_DATA_COLLECTION = "historical_data"  # New collection for cached price data

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

# ========== Historical Data Caching Functions ==========

def save_historical_data_to_db(symbol, df):
    """Save historical price data to MongoDB cache
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        df: DataFrame with columns: timestamp, open, high, low, close, volume
    
    Returns:
        bool: True if successful, False otherwise
    """
    db = get_db_connection()
    if db is None:
        print(f"⚠️ Cannot save {symbol} data: MongoDB not connected")
        return False
    
    if df is None or len(df) == 0:
        print(f"⚠️ No data to save for {symbol}")
        return False
    
    try:
        collection = db[HISTORICAL_DATA_COLLECTION]
        
        # Convert DataFrame to list of documents
        records = []
        for _, row in df.iterrows():
            records.append({
                "symbol": symbol,
                "timestamp": int(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "updated_at": datetime.utcnow()
            })
        
        # Delete existing data for this symbol
        collection.delete_many({"symbol": symbol})
        
        # Insert new data
        if records:
            collection.insert_many(records)
            print(f"✅ Saved {len(records)} records for {symbol} to MongoDB cache")
            return True
        else:
            print(f"⚠️ No records to insert for {symbol}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to save {symbol} data to MongoDB: {e}")
        return False

def get_historical_data_from_db(symbol, min_timestamp=None):
    """Retrieve historical price data from MongoDB cache
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        min_timestamp: Optional minimum timestamp (in milliseconds) to filter data
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, or None if not found
    """
    db = get_db_connection()
    if db is None:
        print(f"⚠️ Cannot fetch {symbol} data: MongoDB not connected")
        return None
    
    try:
        collection = db[HISTORICAL_DATA_COLLECTION]
        
        # Build query
        query = {"symbol": symbol}
        if min_timestamp:
            query["timestamp"] = {"$gte": min_timestamp}
        
        # Fetch data
        cursor = collection.find(query).sort("timestamp", 1)
        records = list(cursor)
        
        if not records:
            print(f"ℹ️ No cached data found for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        print(f"✅ Retrieved {len(df)} cached records for {symbol} from MongoDB")
        return df
        
    except Exception as e:
        print(f"❌ Failed to fetch {symbol} data from MongoDB: {e}")
        return None

def get_last_cached_timestamp(symbol):
    """Get the timestamp of the most recent cached data for a symbol
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
    
    Returns:
        int: Timestamp in milliseconds, or None if no data found
    """
    db = get_db_connection()
    if db is None:
        return None
    
    try:
        collection = db[HISTORICAL_DATA_COLLECTION]
        latest = collection.find_one(
            {"symbol": symbol},
            sort=[("timestamp", -1)]
        )
        
        if latest:
            return int(latest["timestamp"])
        return None
        
    except Exception as e:
        print(f"❌ Failed to get last timestamp for {symbol}: {e}")
        return None

def update_recent_data_in_cache(symbol, df):
    """Update only recent data (last 1-2 days) in MongoDB cache
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        df: DataFrame with new data to add/update
    
    Returns:
        bool: True if successful, False otherwise
    """
    db = get_db_connection()
    if db is None:
        return False
    
    if df is None or len(df) == 0:
        return False
    
    try:
        collection = db[HISTORICAL_DATA_COLLECTION]
        
        # Get timestamps from new data
        new_timestamps = set(df["timestamp"].astype(int).tolist())
        
        # Delete existing records with these timestamps
        collection.delete_many({
            "symbol": symbol,
            "timestamp": {"$in": list(new_timestamps)}
        })
        
        # Insert new/updated records
        records = []
        for _, row in df.iterrows():
            records.append({
                "symbol": symbol,
                "timestamp": int(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "updated_at": datetime.utcnow()
            })
        
        if records:
            collection.insert_many(records)
            print(f"✅ Updated {len(records)} recent records for {symbol} in MongoDB cache")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Failed to update recent data for {symbol}: {e}")
        return False

# ========== MongoDB Setup & Initialization ==========

def setup_mongodb_indexes():
    """Create indexes for optimal query performance
    
    This should be run once after MongoDB setup to optimize queries.
    Creates indexes on:
    - historical_data: {symbol: 1, timestamp: 1} (compound index)
    - predictions: {symbol: 1, timestamp: -1} (for recent predictions)
    
    Returns:
        bool: True if successful, False otherwise
    """
    db = get_db_connection()
    if db is None:
        print("⚠️ Cannot setup indexes: MongoDB not connected")
        return False
    
    try:
        # Setup indexes for historical_data collection
        historical_collection = db[HISTORICAL_DATA_COLLECTION]
        
        # Compound index for fast queries by symbol and timestamp
        historical_collection.create_index(
            [("symbol", 1), ("timestamp", 1)],
            name="symbol_timestamp_idx",
            background=True
        )
        print("✅ Created index on historical_data: symbol + timestamp")
        
        # Index on updated_at for cache freshness checks
        historical_collection.create_index(
            [("updated_at", -1)],
            name="updated_at_idx",
            background=True
        )
        print("✅ Created index on historical_data: updated_at")
        
        # Setup indexes for predictions collection
        predictions_collection = db[COLLECTION_NAME]
        
        # Index for querying recent predictions by symbol
        predictions_collection.create_index(
            [("symbol", 1), ("timestamp", -1)],
            name="symbol_timestamp_idx",
            background=True
        )
        print("✅ Created index on predictions: symbol + timestamp")
        
        print("🎉 MongoDB indexes setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup MongoDB indexes: {e}")
        return False

def verify_mongodb_connection():
    """Verify MongoDB connection and database access
    
    Returns:
        dict: Connection status and database info
    """
    db = get_db_connection()
    if db is None:
        return {
            "connected": False,
            "error": "MONGO_URI not configured or connection failed"
        }
    
    try:
        # Test connection
        db.client.admin.command('ping')
        
        # Get collection counts
        historical_count = db[HISTORICAL_DATA_COLLECTION].count_documents({})
        predictions_count = db[COLLECTION_NAME].count_documents({})
        
        # Get database stats
        db_stats = db.command("dbStats")
        
        return {
            "connected": True,
            "database": DB_NAME,
            "collections": {
                "historical_data": {
                    "count": historical_count,
                    "collection": HISTORICAL_DATA_COLLECTION
                },
                "predictions": {
                    "count": predictions_count,
                    "collection": COLLECTION_NAME
                }
            },
            "database_size_mb": round(db_stats.get("dataSize", 0) / (1024 * 1024), 2),
            "storage_size_mb": round(db_stats.get("storageSize", 0) / (1024 * 1024), 2)
        }
        
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Run setup if executed directly
    print("🔧 MongoDB Setup Script")
    print("=" * 50)
    
    # Verify connection
    print("\n1. Verifying MongoDB connection...")
    status = verify_mongodb_connection()
    if status["connected"]:
        print("✅ MongoDB connection successful!")
        print(f"   Database: {status['database']}")
        print(f"   Historical data records: {status['collections']['historical_data']['count']}")
        print(f"   Prediction records: {status['collections']['predictions']['count']}")
        print(f"   Database size: {status['database_size_mb']} MB")
    else:
        print(f"❌ MongoDB connection failed: {status.get('error', 'Unknown error')}")
        print("\n💡 Make sure MONGO_URI is set in your .env file")
        exit(1)
    
    # Setup indexes
    print("\n2. Setting up indexes...")
    if setup_mongodb_indexes():
        print("✅ Indexes created successfully!")
    else:
        print("⚠️ Some indexes may already exist (this is OK)")
    
    print("\n🎉 MongoDB setup complete!")
