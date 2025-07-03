# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Define input schema for all 24 fields
class InputData(BaseModel):
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

app = FastAPI()

# Load the model
model = joblib.load("app/model.pkl")

@app.get("/")
def root():
    return {"status": "Live", "message": "Welcome to Predictive Maintenance API"}

@app.post("/predict")
def predict(data: InputData):
    input_data = [[
        data.cycle,
        data.op_setting_1,
        data.op_setting_2,
        data.op_setting_3,
        data.sensor_1,
        data.sensor_2,
        data.sensor_3,
        data.sensor_4,
        data.sensor_5,
        data.sensor_6,
        data.sensor_7,
        data.sensor_8,
        data.sensor_9,
        data.sensor_10,
        data.sensor_11,
        data.sensor_12,
        data.sensor_13,
        data.sensor_14,
        data.sensor_15,
        data.sensor_16,
        data.sensor_17,
        data.sensor_18,
        data.sensor_19,
        data.sensor_20,
        data.sensor_21,
    ]]

    prediction = model.predict(input_data)[0]
    return {"prediction": int(prediction)}
