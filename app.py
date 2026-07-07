from fastapi import FastAPI
from typing import Dict, Any

from predictor import predict

from schemas import Passenger

app=FastAPI(title="Titanic Survived API",version="1.0")

@app.get("/")
def home() -> Dict[str, str]:
    return {"Titanic Prediction API is running"}    


@app.post("/predict")
def predict_survival(passenger:Passenger) -> Dict[str, Any]:
    result=predict(passenger.model_dump())
    return result
