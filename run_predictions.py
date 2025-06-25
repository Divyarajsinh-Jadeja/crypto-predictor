# run_predictions.py

import requests
import datetime

# Call all predictions
response = requests.get("http://localhost:5050/predict_all_lstm")
print(response.json())
print(f"Run_predictions.py started at {datetime.datetime.now()}")
