from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("model.pkl")
app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}