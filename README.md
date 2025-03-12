# Poetry를 이용한 멀티프로젝트 가상환경 생성 튜토리얼
- [참고링크](https://techblog.lycorp.co.jp/ko/python-multi-project-application-with-poetry)

## Prerequisite
- Python 3.10.12
- Poetry 2.1.1 이상
```
# poetry 설치
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
```

## 진행 순서
1. `poetry config virtualenvs.in-project true`, `poetry init` (python version: ^3.10)
2. requirements.txt 파일에 버전표기 없이 필요한 패키지 작성
3. pytorch_env 폴더에서 아래 명령어 실행
```
poetry init
poetry env use python3.10
poetry add $(cat requirements.txt)

# 만약 dependency 충돌이 있을 경우, 필요 우선순위가 높은 라이브러리부터 하나씩 `poetry add`로 하나씩 추가하면서 버전을 조절합니다.

# 만약 poetry add로 설치가 되지 않는 라이브러리일 경우
poetry run pip install 라이브러리명
poetry add 라이브러리명 --lock
```

4. 테스트코드 실행
  - `poetry run python tensorflow_test.py` or
  - `source .venv/bin/activate` -> `python tensorflow_test.py`
5. 라이브러리 설치 및 테스트 코드 동작을 확인하면 `poetry run pip freeze > requirements-lock.txt` 로 버전을 고정합니다.
