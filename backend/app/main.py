# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import pandas as pd
from app.predict import load_model, predict

app = FastAPI(title="AI4I Predictive Maintenance API")

# 모델 로드 (서버 시작 시 1번만)
model = load_model(model_path="app/artifacts/model.pth", input_dim=12)

# 요청용 데이터 모델 정의
class PredictRequest(BaseModel):
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: float
    Type: str  # A~G

@app.get("/")
def root():
    return {"message": "AI4I Predictive Maintenance API is running."}

@app.post("/predict")
def predict_endpoint(samples: List[PredictRequest]):
    # 리스트 -> DataFrame
    data = pd.DataFrame([sample.dict() for sample in samples])
    # 예측
    probs = predict(data, model)
    return {"predictions": probs}
