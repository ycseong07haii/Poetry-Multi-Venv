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