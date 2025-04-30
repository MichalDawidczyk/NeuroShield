import tensorflow as tf
import numpy as np
import pickle
import pandas as pd

xgb_model = pickle.load(open("models/xgb_cassifier.pkl", "rb"))
rf_model = pickle.load(open("models/random_forest_classifier.pkl", "rb"))
nn_model = tf.keras.models.load_model("models/neural_network.keras")

def convert_timestamp_to_time(timestamp):
    first_transaction_time = pd.to_datetime("2025-04-30 00:00:00")
    transaction_time = pd.to_datetime(timestamp)
    return (transaction_time - first_transaction_time).total_seconds()

def predict_fraud(transaction):
    default_values = [0.0] * 28
    # transaction_data = transaction.model_dump()
    time = convert_timestamp_to_time(transaction.time)
    features = np.array([transaction.amount, time] + default_values, dtype=np.float32).reshape(1, -1)
    
    xgb_pred = xgb_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    nn_pred = nn_model.predict(features)[0][0]

    return {
        "XGBoost_Prediction": bool(int(xgb_pred)),
        "RandomForest_Prediction": bool(int(rf_pred)),
        "NeuralNetwork_Probability": round(float(nn_pred), 4)
    }
