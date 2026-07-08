import joblib
import pandas as pd



def predict(data: dict):
    
    prediction = 1  # 1 = Survival, 0 = No Survival
    probability = [0.2, 0.8]  # 20% No Survival, 80% Survival
    
    return {
        "prediction": int(prediction),
        "probability": {
            "No_Survival": probability[0],
            "Survival": probability[1]
        }
    }
