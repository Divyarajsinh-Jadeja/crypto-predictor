# classifier_training.py
import os  # Add this import
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from train_model import fetch_klines, add_features

def label_success(df):
    df["target"] = df["close"].shift(-1)
    df["success"] = (df["target"] > df["close"]).astype(int)  # 1 if predicted up and it did
    return df.dropna()

def build_classifier_dataset(symbol="BTCUSDT"):
    df = fetch_klines(symbol)
    df = add_features(df)
    df = label_success(df)
    features = df[["close", "volume", "rsi", "ema7", "momentum"]]
    target = df["success"]
    return features, target

def train_classifier(symbol="BTCUSDT"):
    X, y = build_classifier_dataset(symbol)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{symbol} Classifier Accuracy:", acc)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{symbol}_classifier.pkl")

if __name__ == "__main__":
    top_coins = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    for symbol in top_coins:
        train_classifier(symbol)