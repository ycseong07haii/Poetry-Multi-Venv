"""PyTorch 서비스 컨테이너 정의"""

from dependency_injector import containers, providers
from pathlib import Path
import sys

# common 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.common.containers import AppContainer

from .models import MNISTModel
from .services import PyTorchInferenceService


class PyTorchContainer(AppContainer):
    """PyTorch 서비스 컨테이너"""
    
    # 기본 설정 확장
    config = providers.Configuration()
    
    # AppContainer의 설정 상속
    config.from_dict({
        "model_path": str(Path.cwd() / "models"),
        "output_dir": str(Path.cwd() / "output"),
        "device": "cuda",  # GPU 사용
    })
    
    # 모델 제공자
    mnist_model = providers.Factory(
        MNISTModel,
        device=config.device,
    )
    
    # 서비스 제공자
    inference_service = providers.Factory(
        PyTorchInferenceService,
        model=mnist_model,
        output_dir=config.output_dir,
    ) 