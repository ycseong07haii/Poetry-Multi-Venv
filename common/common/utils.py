"""공통 유틸리티 함수"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from .models import InferenceResult, ReportData


def save_inference_result(result: InferenceResult, output_dir: str) -> str:
    """추론 결과를 JSON 파일로 저장합니다.
    
    Args:
        result: 저장할 추론 결과
        output_dir: 결과를 저장할 디렉토리 경로
        
    Returns:
        저장된 파일의 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.framework}_{result.model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    
    return filepath


def load_inference_results(input_dir: str) -> List[InferenceResult]:
    """디렉토리에서 모든 추론 결과를 로드합니다.
    
    Args:
        input_dir: 추론 결과가 저장된 디렉토리 경로
        
    Returns:
        InferenceResult 객체 리스트
    """
    results = []
    
    for file in Path(input_dir).glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            results.append(InferenceResult(**data))
    
    return results


def create_report(title: str, inference_results: List[InferenceResult], 
                 summary: Dict[str, Any] = None) -> ReportData:
    """추론 결과를 바탕으로 보고서 데이터를 생성합니다.
    
    Args:
        title: 보고서 제목
        inference_results: 추론 결과 리스트
        summary: 요약 정보 (선택 사항)
        
    Returns:
        ReportData 객체
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return ReportData(
        title=title,
        timestamp=timestamp,
        inference_results=inference_results,
        summary=summary or {}
    ) 