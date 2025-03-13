"""PyTorch 서비스 메인 실행 파일"""

import numpy as np
import torch
from pathlib import Path
import sys
import os

# 현재 디렉토리를 모듈 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from pytorch_service.containers import PyTorchContainer


def generate_sample_data() -> np.ndarray:
    """샘플 MNIST 데이터 생성
    
    Returns:
        무작위 MNIST 형태의 데이터
    """
    # 28x28 크기의 무작위 이미지 생성 (단일 채널)
    return np.random.rand(28, 28).astype(np.float32)


def main():
    """메인 함수"""
    # 컨테이너 초기화
    container = PyTorchContainer()
    
    # 환경 변수에서 출력 디렉토리 가져오기
    output_dir = os.environ.get("APP_OUTPUT_DIR", container.config.output_dir())
    
    # 출력 디렉토리 생성
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    
    # 서비스 가져오기
    inference_service = container.inference_service()
    
    # 샘플 데이터 생성
    sample_data = generate_sample_data()
    
    # 추론 실행
    print("PyTorch 모델로 추론을 실행합니다...")
    result = inference_service.run_inference(sample_data)
    
    print(f"추론 결과: {result['result'].output}")
    print(f"결과가 저장된 경로: {result['output_path']}")
    
    return result


if __name__ == "__main__":
    main() 