import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import sqlite3
from transactions import initialize_db
import os
import random

DB_PATH = "transactions.db"

xgb_model = pickle.load(open("models/xgb_cassifier.pkl", "rb"))
rf_model = pickle.load(open("models/random_forest_classifier.pkl", "rb"))
nn_model = tf.keras.models.load_model("models/neural_network.keras")

def convert_timestamp_to_time(timestamp):
    transaction_time = pd.to_datetime(timestamp)
    return transaction_time.timestamp()

def db_check(user_id, location, time):
    if not os.path.exists(DB_PATH):
        initialize_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transactions (user_id, location, time) VALUES (?, ?, ?)
        """, (user_id, location, time))
        conn.commit()
        conn.close()
        return {
            "is_location_suspicious": False,
            "time": time
            }
    else:
        conn = sqlite3.connect("transactions.db")
        cursor = conn.cursor()
        cursor.execute("SELECT location, time FROM transactions WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        if result:
            last_location, last_time = result
            time_to_sec = convert_timestamp_to_time(time)
            last_time_to_sec = convert_timestamp_to_time(last_time)
            if last_location != location and (time_to_sec - last_time_to_sec) < 600:
                conn.close()
                return {
                    "is_location_suspicious": True,
                    "time": result[1]
                    }
        cursor.execute("REPLACE INTO transactions (user_id, location, time) VALUES (?, ?, ?)", (user_id, location, time))
        conn.commit()
        conn.close()
        return {
            "is_location_suspicious": False,
            "time": result[1]
            }

def predict_fraud(transaction):
    default_values = [random.uniform(-5.000, 5.000) for _ in range(28)]
    db_result = db_check(transaction.user_id, transaction.location, transaction.time)
    time = convert_timestamp_to_time(db_result["time"])
    features = np.array([transaction.amount, time] + default_values, dtype=np.float32).reshape(1, -1)

    is_location_suspicious = db_result["is_location_suspicious"]

    xgb_pred = xgb_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    nn_pred = nn_model.predict(features)[0][0]

    return {
        "XGBoost": {"fraud_detected": bool(int(xgb_pred)), "message": "⚠️ Possible Fraud Detected!" if bool(int(xgb_pred)) else "✅ No fraud detected."},
        "RandomForest": {"fraud_detected": bool(int(rf_pred)), "message": "🚨 Fraud Alert!" if bool(int(rf_pred)) else "✅ Transaction appears normal."},
        "NeuralNetwork": {"fraud_probability": round(float(nn_pred), 4), "message": f"Fraud Probability: {round(float(nn_pred) * 100, 2)}%"},
        "LocationCheck": {"risky_location": is_location_suspicious, "message": "🌍 Suspicious transaction location detected!" if is_location_suspicious else "📍 Location is safe."}
    }