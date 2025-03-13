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