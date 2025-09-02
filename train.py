import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# 모델 정의
# -------------------------------
class PredictiveMaintenanceModel(nn.Module):
    """
    간단한 Feedforward Neural Network
    입력 -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    이진분류를 위해 마지막 활성화 함수 Sigmoid 사용
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# 데이터 로드
# -------------------------------
def load_data():
    """
    전처리 완료된 CSV 데이터를 불러와 Tensor로 변환 후 DataLoader 생성
    train_loader와 test_loader 반환
    """
    X_train = pd.read_csv("data/processed/X_train.csv").astype(float).values
    X_test = pd.read_csv("data/processed/X_test.csv").astype(float).values
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Tensor 변환 (모델 입력용)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape 맞춤
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # DataLoader 생성 (배치 단위 학습)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

    return train_loader, test_loader

# -------------------------------
# 학습 함수
# -------------------------------
def train_model(model, train_loader, criterion, optimizer, device):
    """
    한 에포크 동안 모델 학습 수행
    """
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()          # 이전 gradient 초기화
        loss = criterion(model(X_batch), y_batch)  # forward + loss 계산
        loss.backward()                # backward propagation
        optimizer.step()               # 가중치 업데이트

# -------------------------------
# 평가 함수
# -------------------------------
def evaluate_model(model, test_loader, device):
    """
    테스트 데이터에 대한 모델 평가
    Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix 계산
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = outputs.cpu().numpy()
            preds = (outputs >= 0.5).cpu().numpy()  # 0.5 기준 이진분류
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    # 리스트 형태 정리
    y_true = [int(v[0]) for v in y_true]
    y_pred = [int(v[0]) for v in y_pred]
    y_prob = [float(v[0]) for v in y_prob]

    # 평가 지표 계산
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
            "roc_auc": roc_auc, "confusion_matrix": cm.tolist(), "fpr": fpr.tolist(), "tpr": tpr.tolist()}

# -------------------------------
# 시각화 저장
# -------------------------------
def save_visualizations(metrics, save_dir="artifacts"):
    """
    Confusion Matrix와 ROC Curve 시각화 후 저장
    """
    os.makedirs(save_dir, exist_ok=True)

    # Confusion Matrix 시각화
    cm = metrics["confusion_matrix"]
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i][j], ha="center", va="center", color="red")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # ROC Curve 시각화
    fpr, tpr = metrics["fpr"], metrics["tpr"]
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()

# -------------------------------
# 메인 실행
# -------------------------------
def main():
    # MPS(macOS GPU) 또는 CPU 사용
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # 데이터 로드
    train_loader, test_loader = load_data()
    input_dim = pd.read_csv("data/processed/X_train.csv").shape[1]

    # 모델, 손실 함수, 최적화기 정의
    model = PredictiveMaintenanceModel(input_dim).to(device)
    criterion = nn.BCELoss()              # 이진분류용 손실
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 (50 epochs)
    epochs = 50
    for epoch in range(epochs):
        train_model(model, train_loader, criterion, optimizer, device)
        if (epoch+1) % 5 == 0:  # 5 epoch마다 진행 상황 출력
            print(f"Epoch {epoch+1}/{epochs} completed.")

    # 평가
    metrics = evaluate_model(model, test_loader, device)

    # 모델 및 지표 저장
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pth")  # 학습된 모델 저장
    with open("artifacts/metrics.json", "w") as f:           # 평가 지표 저장
        json.dump(metrics, f, indent=4)

    # 시각화 저장
    save_visualizations(metrics)
    print("모델 학습, 평가, 시각화, 저장 완료!")

if __name__ == "__main__":
    main()
