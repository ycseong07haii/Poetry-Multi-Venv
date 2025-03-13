"""공통 데이터 모델 정의"""

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