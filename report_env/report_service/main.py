"""Report 서비스 메인 실행 파일"""

import os
from pathlib import Path
import sys

# 현재 디렉토리를 모듈 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from report_service.containers import ReportContainer


def main():
    """메인 함수"""
    # 컨테이너 초기화
    container = ReportContainer()
    
    # 템플릿 디렉토리 생성
    template_dir = Path(container.config.template_dir())
    template_dir.mkdir(exist_ok=True, parents=True)
    
    # 출력 디렉토리 생성
    output_dir = Path(container.config.output_dir())
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 서비스 가져오기
    report_service = container.report_service()
    
    # 보고서 생성
    print("추론 결과 보고서를 생성합니다...")
    report_path = report_service.generate_report("MNIST 모델 추론 결과 보고서")
    
    if report_path:
        print(f"보고서가 생성되었습니다: {report_path}")
    else:
        print("보고서 생성에 실패했습니다.")
    
    return report_path


if __name__ == "__main__":
    main() 