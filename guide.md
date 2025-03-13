# Poetry와 Dependency Injector를 사용한 멀티서비스 관리 레포 만들기

이 튜토리얼에서는 Python 프로젝트에서 Poetry와 Dependency Injector를 활용하여 여러 독립적인 서비스를 관리하는 방법을 배웁니다. 특히 PyTorch와 TensorFlow와 같이 의존성 충돌이 발생할 수 있는 라이브러리를 사용하는 서비스들을 효과적으로 관리하는 방법을 알아봅니다. **이 레포의 파일들과 가이드는 claude-3.7-sonnet 모델을 통해 작성되었습니다.**

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [프로젝트 구조](#프로젝트-구조)
3. [Poetry를 사용한 가상환경 관리](#poetry를-사용한-가상환경-관리)
4. [Dependency Injector 이해하기](#dependency-injector-이해하기)
5. [공통 모듈 구현](#공통-모듈-구현)
6. [서비스 구현](#서비스-구현)
7. [프로젝트 설정 및 실행](#프로젝트-설정-및-실행)
8. [확장 및 응용 방법](#확장-및-응용-방법)

## 프로젝트 개요

이 프로젝트는 다음과 같은 세 가지 독립적인 서비스로 구성됩니다:

1. **PyTorch 서비스**: PyTorch를 사용하여 MNIST 이미지 분류 모델을 실행합니다.
2. **TensorFlow 서비스**: TensorFlow를 사용하여 MNIST 이미지 분류 모델을 실행합니다.
3. **Report 서비스**: 위 두 서비스의 추론 결과를 수집하여 보고서를 생성합니다.

각 서비스는 독립적인 가상환경에서 실행되지만, 공통 모듈을 통해 데이터를 공유합니다.

## 프로젝트 구조

프로젝트는 다음과 같은 구조로 구성됩니다:

```
project/
├── common/                  # 공통 모듈
│   ├── common/              # 공통 코드
│   │   ├── __init__.py
│   │   ├── containers.py    # 기본 DI 컨테이너
│   │   ├── models.py        # 공통 데이터 모델
│   │   └── utils.py         # 유틸리티 함수
│   ├── pyproject.toml       # Poetry 설정
│   └── poetry.lock          # 의존성 잠금 파일
├── pytorch_env/             # PyTorch 서비스 환경
│   ├── pytorch_service/     # PyTorch 서비스 코드
│   │   ├── __init__.py
│   │   ├── containers.py    # PyTorch DI 컨테이너
│   │   ├── main.py          # 진입점
│   │   ├── models.py        # PyTorch 모델
│   │   └── services.py      # 서비스 구현
│   ├── pyproject.toml       # Poetry 설정
│   └── requirements.txt     # 필요한 패키지 목록
├── tensorflow_env/          # TensorFlow 서비스 환경
│   ├── tensorflow_service/  # TensorFlow 서비스 코드
│   │   ├── __init__.py
│   │   ├── containers.py    # TensorFlow DI 컨테이너
│   │   ├── main.py          # 진입점
│   │   ├── models.py        # TensorFlow 모델
│   │   └── services.py      # 서비스 구현
│   ├── pyproject.toml       # Poetry 설정
│   └── requirements.txt     # 필요한 패키지 목록
├── report_env/              # Report 서비스 환경
│   ├── report_service/      # Report 서비스 코드
│   │   ├── __init__.py
│   │   ├── containers.py    # Report DI 컨테이너
│   │   ├── main.py          # 진입점
│   │   └── services.py      # 서비스 구현
│   ├── templates/           # 보고서 템플릿
│   ├── reports/             # 생성된 보고서
│   ├── pyproject.toml       # Poetry 설정
│   └── requirements.txt     # 필요한 패키지 목록
├── output/                  # 공통 출력 디렉토리
├── setup.sh                 # 환경 설정 스크립트
└── run.sh                   # 실행 스크립트
```

## Poetry를 사용한 가상환경 관리

[Poetry](https://python-poetry.org/)는 Python 프로젝트의 의존성 관리와 패키징을 위한 도구입니다. 이 프로젝트에서는 Poetry를 사용하여 각 서비스별로 독립적인 가상환경을 생성하고 관리합니다.

### Poetry 설치

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Poetry 기본 명령어

- 프로젝트 초기화: `poetry init`
- 의존성 설치: `poetry install`
- 패키지 추가: `poetry add <패키지명>`
- 가상환경에서 명령 실행: `poetry run <명령어>`
- 가상환경 쉘 진입: `poetry shell`

### 프로젝트 내 가상환경 설정

Poetry는 기본적으로 중앙 위치에 가상환경을 생성하지만, 프로젝트 디렉토리 내에 가상환경을 생성하도록 설정할 수 있습니다:

```bash
poetry config virtualenvs.in-project true
```

## Dependency Injector 이해하기

[Dependency Injector](https://python-dependency-injector.ets-labs.org/)는 Python용 의존성 주입 프레임워크입니다. 이 프로젝트에서는 Dependency Injector를 사용하여 서비스 간의 의존성을 관리하고 코드의 결합도를 낮춥니다.

### 의존성 주입이란?

의존성 주입(Dependency Injection)은 객체가 필요로 하는 의존성을 외부에서 주입받는 디자인 패턴입니다. 이를 통해:

1. 코드의 결합도를 낮출 수 있습니다.
2. 테스트 용이성이 향상됩니다.
3. 코드의 재사용성이 증가합니다.
4. 관심사의 분리가 가능해집니다.

### Dependency Injector의 주요 개념

#### 1. 컨테이너(Container)

컨테이너는 의존성 주입을 관리하는 중앙 객체입니다. 이 프로젝트에서는 각 서비스마다 컨테이너를 정의하여 필요한 객체들을 관리합니다.

```python
from dependency_injector import containers, providers

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
```

#### 2. 제공자(Provider)

제공자는 객체를 생성하고 의존성을 주입하는 역할을 합니다. 주요 제공자 유형은 다음과 같습니다:

- **Factory**: 호출될 때마다 새 객체를 생성합니다.
- **Singleton**: 첫 호출 시 객체를 생성하고 이후 호출에서는 동일한 객체를 반환합니다.
- **Configuration**: 설정 값을 관리합니다.
- **Resource**: 리소스(예: 데이터베이스 연결)를 관리합니다.

```python
# 모델 제공자 (Factory)
mnist_model = providers.Factory(
    MNISTModel,
    device=config.device,
)

# 서비스 제공자 (Factory)
inference_service = providers.Factory(
    PyTorchInferenceService,
    model=mnist_model,
    output_dir=config.output_dir,
)
```

#### 3. 와이어링(Wiring)

와이어링은 의존성 주입을 함수나 클래스에 적용하는 방법입니다. `@inject` 데코레이터를 사용하여 의존성을 주입받을 수 있습니다.

```python
from dependency_injector.wiring import inject, Provide

@inject
def main(
    inference_service: PyTorchInferenceService = Provide[PyTorchContainer.inference_service]
):
    # inference_service 사용
    pass
```

## 공통 모듈 구현

공통 모듈은 모든 서비스에서 공유하는 코드를 포함합니다. 이 프로젝트에서는 다음과 같은 공통 모듈을 구현합니다:

### 1. 데이터 모델 (models.py)

```python
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class InferenceResult(BaseModel):
    """추론 결과를 저장하는 모델"""
    model_name: str
    framework: str
    input_data: Any
    output: Any
    metadata: Optional[Dict[str, Any]] = None

class ReportData(BaseModel):
    """보고서 데이터 모델"""
    title: str
    timestamp: str
    inference_results: List[InferenceResult]
    summary: Optional[Dict[str, Any]] = None
```

### 2. 유틸리티 함수 (utils.py)

```python
def save_inference_result(result: InferenceResult, output_dir: str) -> str:
    """추론 결과를 JSON 파일로 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.framework}_{result.model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    
    return filepath

def load_inference_results(input_dir: str) -> List[InferenceResult]:
    """디렉토리에서 모든 추론 결과를 로드합니다."""
    results = []
    
    for file in Path(input_dir).glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            results.append(InferenceResult(**data))
    
    return results

def create_report(title: str, inference_results: List[InferenceResult], 
                 summary: Dict[str, Any] = None) -> ReportData:
    """추론 결과를 바탕으로 보고서 데이터를 생성합니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return ReportData(
        title=title,
        timestamp=timestamp,
        inference_results=inference_results,
        summary=summary or {}
    )
```

### 3. 기본 컨테이너 (containers.py)

```python
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
```

## 서비스 구현

각 서비스는 독립적인 가상환경에서 실행되며, 공통 모듈을 통해 데이터를 공유합니다.

### 1. PyTorch 서비스

#### 컨테이너 (containers.py)

```python
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
```

#### 메인 실행 파일 (main.py)

```python
import numpy as np
import torch
from pathlib import Path
import sys
import os

# 현재 디렉토리를 모듈 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from pytorch_service.containers import PyTorchContainer

def generate_sample_data() -> np.ndarray:
    """샘플 MNIST 데이터 생성"""
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
```

### 2. TensorFlow 서비스

TensorFlow 서비스는 PyTorch 서비스와 유사한 구조를 가지지만, TensorFlow 라이브러리를 사용합니다.

### 3. Report 서비스

Report 서비스는 PyTorch와 TensorFlow 서비스의 추론 결과를 수집하여 보고서를 생성합니다.

#### 컨테이너 (containers.py)

```python
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
```

## 프로젝트 설정 및 실행

### 1. 환경 설정 (setup.sh)

```bash
#!/bin/bash

# Poetry 설정: 프로젝트 내에 가상환경 생성
poetry config virtualenvs.in-project true

# 공통 모듈 설치
echo "공통 모듈을 설치합니다..."
cd common
poetry init -n
poetry env use python3.10
poetry install
cd ..

# PyTorch 환경 설치
echo "PyTorch 환경을 설치합니다..."
cd pytorch_env
poetry init -n
poetry env use python3.10
poetry add $(cat requirements.txt)
cd ..

# TensorFlow 환경 설치
echo "TensorFlow 환경을 설치합니다..."
cd tensorflow_env
poetry init -n
poetry env use python3.10
poetry add $(cat requirements.txt)
cd ..

# Report 환경 설치
echo "Report 환경을 설치합니다..."
cd report_env
poetry init -n
poetry env use python3.10
poetry add $(cat requirements.txt)
cd ..

echo "모든 환경 설치가 완료되었습니다."
```

### 2. 서비스 실행 (run.sh)

```bash
#!/bin/bash

# 공통 출력 디렉토리 생성
COMMON_OUTPUT_DIR="$(pwd)/output"
mkdir -p "$COMMON_OUTPUT_DIR"
export APP_OUTPUT_DIR="$COMMON_OUTPUT_DIR"

echo "공통 출력 디렉토리: $COMMON_OUTPUT_DIR"

# PyTorch 서비스 실행
echo "PyTorch 서비스를 실행합니다..."
cd pytorch_env
APP_OUTPUT_DIR="$COMMON_OUTPUT_DIR" poetry run python -m pytorch_service.main
cd ..

# TensorFlow 서비스 실행
echo "TensorFlow 서비스를 실행합니다..."
cd tensorflow_env
APP_OUTPUT_DIR="$COMMON_OUTPUT_DIR" poetry run python -m tensorflow_service.main
cd ..

# Report 서비스 실행
echo "Report 서비스를 실행합니다..."
cd report_env
APP_INPUT_DIR="$COMMON_OUTPUT_DIR" poetry run python -m report_service.main
cd ..

echo "모든 서비스 실행이 완료되었습니다."
```

## Dependency Injector 심층 이해

### 1. 컨테이너 상속

이 프로젝트에서는 `AppContainer`라는 기본 컨테이너를 정의하고, 각 서비스별 컨테이너가 이를 상속받아 확장합니다. 이를 통해 공통 설정과 기능을 재사용할 수 있습니다.

```python
class PyTorchContainer(AppContainer):
    # AppContainer의 모든 기능을 상속받고 추가 기능 구현
    pass
```

### 2. 설정 관리

Dependency Injector의 `Configuration` 제공자를 사용하면 다양한 소스(딕셔너리, 환경 변수, 파일 등)에서 설정을 로드하고 관리할 수 있습니다.

```python
# 기본 설정 정의
config.set_default_values({
    "output_dir": str(Path.cwd() / "output"),
})

# 환경 변수에서 설정 로드 (APP_ 접두사 사용)
config.load_from_env("APP_", "app")

# 다른 설정 소스에서 설정 상속
config.from_dict({
    "model_path": str(Path.cwd() / "models"),
})
```

### 3. 객체 생성 및 의존성 주입

Dependency Injector는 객체 생성과 의존성 주입을 자동화합니다. 예를 들어, `inference_service` 제공자는 `PyTorchInferenceService` 객체를 생성하고, 필요한 의존성(`model`, `output_dir`)을 자동으로 주입합니다.

```python
inference_service = providers.Factory(
    PyTorchInferenceService,
    model=mnist_model,  # 다른 제공자에서 생성된 객체
    output_dir=config.output_dir,  # 설정 값
)
```

### 4. 런타임 의존성 해결

Dependency Injector는 런타임에 의존성을 해결합니다. 즉, 객체가 실제로 필요할 때까지 생성을 지연시킵니다. 이를 통해 리소스를 효율적으로 사용할 수 있습니다.

```python
# 컨테이너 초기화 (아직 객체 생성 안 됨)
container = PyTorchContainer()

# 서비스 가져오기 (이 시점에 객체 생성 및 의존성 주입)
inference_service = container.inference_service()
```

## 공통 모듈 상세 설명

### 1. models.py

`models.py` 파일은 Pydantic을 사용하여 데이터 모델을 정의합니다. 이 모델들은 서비스 간에 데이터를 교환하는 데 사용됩니다.

- `InferenceResult`: 모델 추론 결과를 저장하는 모델입니다. 모델 이름, 프레임워크, 입력 데이터, 출력 결과, 메타데이터 등을 포함합니다.
- `ReportData`: 보고서 데이터를 저장하는 모델입니다. 제목, 타임스탬프, 추론 결과 목록, 요약 정보 등을 포함합니다.

### 2. utils.py

`utils.py` 파일은 여러 서비스에서 공통으로 사용하는 유틸리티 함수를 제공합니다.

- `save_inference_result`: 추론 결과를 JSON 파일로 저장합니다.
- `load_inference_results`: 디렉토리에서 모든 추론 결과를 로드합니다.
- `create_report`: 추론 결과를 바탕으로 보고서 데이터를 생성합니다.

### 3. containers.py

`containers.py` 파일은 기본 애플리케이션 컨테이너를 정의합니다. 이 컨테이너는 모든 서비스 컨테이너의 기본이 되며, 공통 설정과 기능을 제공합니다.

- `AppContainer`: 기본 설정(출력 디렉토리, 데이터 디렉토리 등)을 정의하고, 환경 변수에서 설정을 로드합니다.

## 확장 및 응용 방법

이 프로젝트는 다양한 방식으로 확장하고 응용할 수 있습니다:

### 1. 새로운 서비스 추가

새로운 서비스를 추가하려면:

1. 새 디렉토리 생성 (예: `new_service_env`)
2. Poetry 환경 설정 (`pyproject.toml`, `requirements.txt`)
3. 서비스 코드 구현 (`containers.py`, `models.py`, `services.py`, `main.py`)
4. `setup.sh`와 `run.sh` 스크립트 업데이트

### 2. 웹 API 통합

FastAPI나 Flask를 사용하여 각 서비스를 웹 API로 노출할 수 있습니다:

```python
from fastapi import FastAPI, Depends
from dependency_injector.wiring import inject, Provide

app = FastAPI()

@app.post("/inference")
@inject
def run_inference(
    data: dict,
    inference_service = Depends(Provide[PyTorchContainer.inference_service])
):
    result = inference_service.run_inference(data["input"])
    return result
```

### 3. 데이터베이스 통합

SQLAlchemy와 같은 ORM을 사용하여 데이터베이스 통합을 추가할 수 있습니다:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class DatabaseContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    db_engine = providers.Singleton(
        create_engine,
        config.db_url,
    )
    
    db_session_factory = providers.Factory(
        sessionmaker,
        bind=db_engine,
    )
```

### 4. 비동기 처리

asyncio를 사용하여 비동기 처리를 추가할 수 있습니다:

```python
import asyncio

class AsyncService:
    async def process_data(self, data):
        # 비동기 처리 로직
        pass

class AsyncContainer(containers.DeclarativeContainer):
    async_service = providers.Factory(AsyncService)
```

## 결론

이 튜토리얼에서는 Poetry와 Dependency Injector를 사용하여 여러 독립적인 서비스를 관리하는 방법을 배웠습니다. 이 접근 방식은 다음과 같은 이점을 제공합니다:

1. **의존성 격리**: 각 서비스는 독립적인 가상환경에서 실행되므로 의존성 충돌이 발생하지 않습니다.
2. **코드 재사용**: 공통 모듈을 통해 코드를 재사용할 수 있습니다.
3. **유연한 설정**: Dependency Injector를 사용하여 설정을 유연하게 관리할 수 있습니다.
4. **테스트 용이성**: 의존성 주입을 통해 단위 테스트가 용이해집니다.
5. **확장성**: 새로운 서비스를 쉽게 추가할 수 있습니다.

이 패턴은 마이크로서비스 아키텍처나 복잡한 ML 파이프라인과 같이 여러 독립적인 컴포넌트로 구성된 시스템에 특히 유용합니다. 