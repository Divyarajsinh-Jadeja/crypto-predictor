#!/usr/bin/env python3
"""
Download and verify trained models.
This script ensures all required model files exist before starting the API.
"""

import os
import sys

def check_models():
    """Check if all model files exist"""
    coins = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "SHIBUSDT", "DOTUSDT",
        "MATICUSDT", "TRXUSDT", "LINKUSDT", "LTCUSDT", "UNIUSDT",
        "BCHUSDT", "XLMUSDT", "NEARUSDT", "ICPUSDT", "APTUSDT"
    ]
    
    missing_models = []
    
    for coin in coins:
        files = [
            f"models/{coin}_lstm_model.h5",
            f"models/{coin}_classifier.pkl",
            f"models/{coin}_prophet.pkl",
            f"models/{coin}scaler.pkl"
        ]
        
        for file in files:
            if not os.path.exists(file):
                missing_models.append(file)
    
    if missing_models:
        print("❌ Missing model files:")
        for model in missing_models:
            print(f"   - {model}")
        return False
    else:
        print(f"✅ All {len(coins) * 4} model files found!")
        return True

def main():
    print("🔍 Checking trained models...")
    
    if not os.path.exists("models"):
        print("📁 Creating models directory...")
        os.makedirs("models", exist_ok=True)
    
    if check_models():
        print("✅ Model verification complete!")
        sys.exit(0)
    else:
        print("\n⚠️  WARNING: Some model files are missing!")
        print("   The API will return 'Insufficient data' errors for missing models.")
        print("   Run 'python train_model.py' to train all models.")
        sys.exit(1)

if __name__ == "__main__":
    main()
