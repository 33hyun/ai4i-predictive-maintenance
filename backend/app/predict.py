import torch
import pandas as pd
from app.model import PredictiveMaintenanceModel
import joblib

# StandardScaler 로드
scaler = joblib.load("app/artifacts/scaler.pkl")

def load_model(model_path="app/artifacts/model.pth", input_dim=5):
    model = PredictiveMaintenanceModel(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_input(df):
    # Type 원-핫 인코딩
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)
    # Numeric feature scaling
    numeric_features = ["Air_temperature", "Process_temperature",
                        "Rotational_speed", "Torque", "Tool_wear"]
    df[numeric_features] = scaler.transform(df[numeric_features])
    return torch.tensor(df.values, dtype=torch.float32)

def predict(df, model):
    X = preprocess_input(df)
    with torch.no_grad():
        probs = model(X).numpy().flatten()
    return probs
