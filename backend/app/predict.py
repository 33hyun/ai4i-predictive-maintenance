# app/predict.py
import torch
import pandas as pd
from .model import PredictiveMaintenanceModel
import joblib

# StandardScaler 로드
scaler = joblib.load("app/artifacts/scaler.pkl")

# 예측용 컬럼 순서
feature_order = [
    "Air_temperature", "Process_temperature", "Rotational_speed",
    "Torque", "Tool_wear", "Type_A", "Type_B", "Type_C", "Type_D",
    "Type_E", "Type_F", "Type_G"
]

def load_model(model_path="app/artifacts/model.pth", input_dim=12):
    model = PredictiveMaintenanceModel(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_input(df: pd.DataFrame):
    # Type 원-핫 인코딩
    df = pd.get_dummies(df, columns=["Type"], drop_first=False)
    
    # 없는 타입 컬럼 생성 (0으로 채움)
    for col in feature_order[5:]:
        if col not in df.columns:
            df[col] = 0

    # 순서 맞추기
    df = df[feature_order]

    # Numeric feature scaling
    numeric_features = feature_order[:5]
    df[numeric_features] = scaler.transform(df[numeric_features])

    return torch.tensor(df.values, dtype=torch.float32)

def predict(df: pd.DataFrame, model):
    X = preprocess_input(df)
    with torch.no_grad():
        probs = model(X).numpy().flatten()
    return probs.tolist()
