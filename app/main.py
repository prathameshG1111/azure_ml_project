from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Define input schema
class InputData(BaseModel):
    cycle: int
    sensor_1: float
    sensor_2: float
    sensor_3: float
    # Add all features used by your model

@app.get("/")
def read_root():
    return {"msg": "Predictive Maintenance API is live!"}

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.cycle, data.sensor_1, data.sensor_2, data.sensor_3]])
    prediction = model.predict(input_array)[0]
    return {"prediction": int(prediction)}
