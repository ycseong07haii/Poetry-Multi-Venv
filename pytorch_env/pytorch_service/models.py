"""PyTorch 모델 정의"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class MNISTModel(nn.Module):
    """간단한 MNIST 분류 모델"""
    
    def __init__(self, device: str = "cuda"):
        """모델 초기화
        
        Args:
            device: 모델을 실행할 디바이스 ('cuda' 또는 'cpu')
        """
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # GPU 사용 가능 여부 확인
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.to(self.device)
        
        print(f"PyTorch 모델이 {self.device} 디바이스에서 실행됩니다.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파
        
        Args:
            x: 입력 텐서 (배치, 채널, 높이, 너비)
            
        Returns:
            출력 텐서 (배치, 클래스 수)
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """예측 수행
        
        Args:
            x: 입력 텐서
            
        Returns:
            예측 결과와 메타데이터
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            output = self.forward(x)
            pred = output.argmax(dim=1)
            probs = F.softmax(output, dim=1)
            
            metadata = {
                "confidence": probs.max(dim=1)[0].cpu().numpy().tolist(),
                "device": self.device,
                "framework": "pytorch",
            }
            
            return pred.cpu(), metadata 