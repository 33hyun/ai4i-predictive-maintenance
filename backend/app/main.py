from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from app.predict import load_model, predict  # predict.py 사용

app = FastAPI(title="Predictive Maintenance API")

# 모델 로드
model = load_model(model_path="app/artifacts/model.pth", input_dim=5)

# 요청 데이터 스키마
class InputData(BaseModel):
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: float
    Type: str

@app.post("/predict")
def get_prediction(data: InputData):
    df = pd.DataFrame([data.dict()])
    prob = predict(df, model)[0]
    return {"failure_probability": float(prob)}
