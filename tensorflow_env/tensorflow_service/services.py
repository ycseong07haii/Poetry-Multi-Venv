"""TensorFlow 추론 서비스"""

import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Any

# common 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.common.models import InferenceResult
from common.common.utils import save_inference_result

from .models import MNISTModel


class TensorFlowInferenceService:
    """TensorFlow 모델 추론 서비스"""
    
    def __init__(self, model: MNISTModel, output_dir: str):
        """서비스 초기화
        
        Args:
            model: 추론에 사용할 TensorFlow 모델
            output_dir: 결과를 저장할 디렉토리 경로
        """
        self.model = model
        # 환경 변수에서 출력 디렉토리 가져오기
        self.output_dir = os.environ.get("APP_OUTPUT_DIR", output_dir)
    
    def run_inference(self, input_data: np.ndarray) -> Dict[str, Any]:
        """추론 실행
        
        Args:
            input_data: 입력 데이터 (numpy 배열)
            
        Returns:
            추론 결과와 메타데이터를 포함한 딕셔너리
        """
        # 추론 실행
        predictions, metadata = self.model.predict(input_data)
        
        # 결과 생성
        result = InferenceResult(
            model_name="MNIST_CNN",
            framework="tensorflow",
            input_data=input_data.shape,
            output=predictions.tolist(),
            metadata=metadata
        )
        
        # 결과 저장
        output_path = save_inference_result(result, self.output_dir)
        
        return {
            "result": result,
            "output_path": output_path
        } 