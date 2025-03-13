"""Dependency Injector 컨테이너 정의"""

from dependency_injector import containers, providers
from pathlib import Path


class AppContainer(containers.DeclarativeContainer):
    """기본 애플리케이션 컨테이너"""
    
    config = providers.Configuration()
    
    # 기본 설정
    config.set_default_values({
        "output_dir": str(Path.cwd() / "output"),
        "data_dir": str(Path.cwd() / "data"),
    })
    
    # 환경 변수에서 설정 로드
    config.load_from_env("APP_", "app") 