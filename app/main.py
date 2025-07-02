# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# ✅ Update model path
model = joblib.load("app/model_clean.pkl")

# ✅ Complete feature set used by your trained model
class SensorInput(BaseModel):
    cycle: int
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

@app.get("/")
def home():
    return {"message": "✅ Predictive Maintenance API is live!"}

@app.post("/predict")
def predict(data: SensorInput):
    features = np.array([list(data.dict().values())])
    prediction = model.predict(features)
    return {"predicted_label": int(prediction[0])}
