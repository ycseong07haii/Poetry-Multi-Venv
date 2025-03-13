# Poetry를 활용한 멀티 프로젝트 Python 애플리케이션

이 프로젝트는 Poetry를 사용하여 여러 가상환경을 관리하고, Dependency Injector를 활용하여 서비스 간 의존성을 관리하는 멀티 프로젝트 Python 애플리케이션입니다.

## 프로젝트 구조

- `common`: 공통 모듈 (모든 서비스에서 사용하는 공통 코드)
- `pytorch_env`: PyTorch 기반 추론 서비스 (GPU 사용)
- `tensorflow_env`: TensorFlow 기반 추론 서비스 (GPU 사용)
- `report_env`: 추론 결과를 종합하여 보고서를 생성하는 서비스

## 요구사항

- Python 3.10
- Poetry (패키지 및 가상환경 관리)
- CUDA (GPU 사용을 위해)

## 설치 방법

1. Poetry 설치 (아직 설치하지 않은 경우)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 프로젝트 클론

```bash
git clone <repository-url>
cd <repository-directory>
```

3. 각 환경 설치

```bash
chmod +x setup.sh
./setup.sh
```

## 실행 방법

모든 서비스를 순차적으로 실행:

```bash
chmod +x run.sh
./run.sh
```

각 서비스를 개별적으로 실행:

```bash
# PyTorch 서비스 실행
cd pytorch_env
poetry run python -m pytorch_service.main

# TensorFlow 서비스 실행
cd tensorflow_env
poetry run python -m tensorflow_service.main

# Report 서비스 실행
cd report_env
poetry run python -m report_service.main
```

## 진행 순서

1. `poetry config virtualenvs.in-project true`, `poetry init` (python version: ^3.10)
2. requirements.txt 파일에 버전표기 없이 필요한 패키지 작성
3. 각 서비스 폴더에서 아래 명령어 실행

```bash
poetry init
poetry env use python3.10
poetry add $(cat requirements.txt)

# 만약 dependency 충돌이 있을 경우, 필요 우선순위가 높은 라이브러리부터 하나씩 `poetry add`로 하나씩 추가하면서 버전을 조절합니다.

# 만약 poetry add로 설치가 되지 않는 라이브러리일 경우
poetry run pip install 라이브러리명
poetry add 라이브러리명 --lock
```

## 프로젝트 특징

1. **독립적인 가상환경**: 각 서비스는 자체 가상환경을 가지고 있어 패키지 충돌 없이 독립적으로 실행됩니다.
2. **공통 모듈 공유**: 모든 서비스는 공통 모듈을 참조하여 코드 중복을 방지합니다.
3. **Dependency Injection**: Dependency Injector를 사용하여 서비스 간 의존성을 관리합니다.
4. **GPU 활용**: PyTorch와 TensorFlow 서비스는 GPU를 활용하여 추론을 수행합니다.
5. **결과 통합**: 각 서비스의 결과를 JSON 파일로 저장하고, Report 서비스에서 이를 종합하여 마크다운 보고서를 생성합니다.
