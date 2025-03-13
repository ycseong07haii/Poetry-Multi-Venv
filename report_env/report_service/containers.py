"""Report 서비스 컨테이너 정의"""

from dependency_injector import containers, providers
from pathlib import Path
import sys
import os

# common 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.common.containers import AppContainer

from .services import ReportService


class ReportContainer(AppContainer):
    """Report 서비스 컨테이너"""
    
    # 기본 설정 확장
    config = providers.Configuration()
    
    # 환경 변수에서 입력 디렉토리 가져오기
    input_dir = os.environ.get("APP_INPUT_DIR", str(Path.cwd().parent / "output"))
    
    # AppContainer의 설정 상속
    config.from_dict({
        "input_dir": input_dir,
        "output_dir": str(Path.cwd() / "reports"),
        "template_dir": str(Path.cwd() / "templates"),
    })
    
    # 서비스 제공자
    report_service = providers.Factory(
        ReportService,
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        template_dir=config.template_dir,
    ) 