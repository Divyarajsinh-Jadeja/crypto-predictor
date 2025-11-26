"""
Script to download models from cloud storage or verify local models exist.
This can be used during Render deployment to ensure models are available.
"""
import os
import requests
import zipfile
import io
from pathlib import Path

MODELS_DIR = "models"
REQUIRED_MODELS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "SHIBUSDT", "DOTUSDT", "MATICUSDT", "TRXUSDT",
    "LINKUSDT", "LTCUSDT", "UNIUSDT", "BCHUSDT", "XLMUSDT", "NEARUSDT",
    "ICPUSDT", "APTUSDT"
]

def check_models_exist():
    """Check if required model files exist"""
    missing = []
    for symbol in REQUIRED_MODELS:
        required_files = [
            f"{MODELS_DIR}/{symbol}_lstm_model.h5",
            f"{MODELS_DIR}/{symbol}scaler.pkl"
        ]
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing.append(file_path)
    return len(missing) == 0, missing

def download_from_url(url, extract_to=MODELS_DIR):
    """Download and extract models from a URL (zip file)"""
    try:
        print(f"📥 Downloading models from {url}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        print("📦 Extracting models...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Models downloaded and extracted successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return False

def main():
    """Main function to check or download models"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    all_exist, missing = check_models_exist()
    
    if all_exist:
        print("✅ All required models are present!")
        return 0
    
    print(f"⚠️ Missing {len(missing)} model files")
    print("Missing files:")
    for f in missing[:10]:  # Show first 10
        print(f"  - {f}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")
    
    # Check if MODELS_URL environment variable is set
    models_url = os.getenv("MODELS_URL")
    if models_url:
        print(f"\n📥 Attempting to download from {models_url}...")
        if download_from_url(models_url):
            # Verify again
            all_exist, _ = check_models_exist()
            if all_exist:
                print("✅ All models downloaded successfully!")
                return 0
            else:
                print("⚠️ Some models still missing after download")
                return 1
        else:
            return 1
    else:
        print("\n💡 To download models automatically, set MODELS_URL environment variable")
        print("   Example: MODELS_URL=https://your-storage.com/models.zip")
        print("\n📝 Alternative: Add models/ directory to git and push to repository")
        return 1

if __name__ == "__main__":
    exit(main())

