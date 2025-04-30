from fastapi import FastAPI
from models_predict import predict_fraud
from schemas import Transaction

app = FastAPI()

@app.post("/predict/")
def predict(transaction: Transaction):
    print("Before calling predict_fraud...")
    result = predict_fraud(transaction)
    print("After calling predict_fraud...")
    return result

@app.get("/healthCheck")
def health_check():
    return {"message": "FastAPI is running!"}