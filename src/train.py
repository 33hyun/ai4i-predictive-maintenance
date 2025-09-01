import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from src.model import PredictiveMaintenanceModel  # 모델 정의 import

# 디바이스 설정: MPS(Mac GPU) 사용 가능하면 사용, 아니면 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# 데이터 불러오기 (전처리 완료 CSV 파일)
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# bool 타입 컬럼을 float로 변환 (PyTorch 텐서 변환을 위해)
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Pandas DataFrame → PyTorch Tensor 변환
# y_train, y_test는 2차원 형태로 변환 (BCELoss 사용 시 필요)
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
y_test = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# 학습용/테스트용 데이터셋 및 DataLoader 생성
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 모델 초기화
model = PredictiveMaintenanceModel(input_dim=X_train.shape[1]).to(device)

# 손실 함수와 최적화 알고리즘 설정
criterion = nn.BCELoss()  # 이진 분류용 손실
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 10
for epoch in range(epochs):
    model.train()  # 학습 모드로 전환
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()         # 기울기 초기화
        outputs = model(X_batch)      # 모델 순전파
        loss = criterion(outputs, y_batch)  # 손실 계산
        loss.backward()               # 역전파
        optimizer.step()              # 파라미터 업데이트
        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# 평가
model.eval()  # 평가 모드
with torch.no_grad():  # 기울기 계산 X
    y_pred = model(X_test.to(device))
    y_pred_label = (y_pred.cpu() >= 0.5).float()  # 0.5 기준으로 클래스 분류
    accuracy = (y_pred_label == y_test).float().mean()
print(f"Test Accuracy: {accuracy:.4f}")
