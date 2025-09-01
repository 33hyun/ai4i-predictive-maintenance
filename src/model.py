import torch
import torch.nn as nn


class PredictiveMaintenanceModel(nn.Module):
    """
    Predictive Maintenance Model
    ----------------------------
    입력: sensor/operational features
    출력: 고장 여부 (0=정상, 1=고장)
    """

    def __init__(self, input_dim: int):
        super(PredictiveMaintenanceModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
            nn.Sigmoid()   # 이진 분류 확률 출력
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (Forward propagation)
        """
        return self.model(x)
