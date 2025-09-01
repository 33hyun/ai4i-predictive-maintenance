import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(csv_path="data/raw/predictive_maintenance.csv"):
    df = pd.read_csv(csv_path)
    
    # 불필요 컬럼 제거
    df = df.drop(columns=["UDI", "Product ID"])
    
    # 범주형 변수   원-핫 인코딩
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)
    
    # 입력/타겟 분리
    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 수치형 스케일링
    numeric_features = ["Air temperature [K]", "Process temperature [K]",
                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # 전처리 완료 데이터 반환
    return X_train, X_test, y_train, y_test
