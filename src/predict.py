# src/predict.py
import torch
import pandas as pd
from src.model import PredictiveMaintenanceModel
from sklearn.preprocessing import StandardScaler

def load_model(model_path="artifacts/model.pth", input_dim=10):
    model = PredictiveMaintenanceModel(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_input(df):
    # 불필요 컬럼 제거
    df = df.drop(columns=["UDI", "Product ID"], errors='ignore')
    
    # Type 원-핫 인코딩
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)
    
    # 학습 때 사용한 컬럼 순서 맞추기
    numeric_features = ["Air temperature [K]", "Process temperature [K]",
                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return torch.tensor(df.values, dtype=torch.float32)

def predict(df, model):
    X = preprocess_input(df)
    with torch.no_grad():
        probs = model(X).numpy().flatten()
    return probs

if __name__ == "__main__":
    # 테스트용
    test_df = pd.read_csv("data/raw/ai4i2020.csv").head(5)  # 첫 5개 샘플
    model = load_model(input_dim=test_df.shape[1]-2)  # UDI, ProductID 제거
    probs = predict(test_df, model)
    for i, p in enumerate(probs):
        print(f"Sample {i}: 고장 확률 = {p:.4f}")
