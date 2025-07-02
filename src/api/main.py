
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
from .pydantic_models import CustomerData

app = FastAPI()

# Load model
model = mlflow.sklearn.load_model("models:/RiskModel/Production")

@app.post("/predict")
async def predict(data: CustomerData):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": "high" if prediction == 1 else "low"
    }
