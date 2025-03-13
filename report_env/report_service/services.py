"""Report 생성 서비스"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import markdown
import jinja2

# common 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.common.models import InferenceResult, ReportData
from common.common.utils import load_inference_results


class ReportService:
    """추론 결과 보고서 생성 서비스"""
    
    def __init__(self, input_dir: str, output_dir: str, template_dir: str):
        """서비스 초기화
        
        Args:
            input_dir: 추론 결과가 저장된 디렉토리 경로
            output_dir: 보고서를 저장할 디렉토리 경로
            template_dir: 템플릿 파일이 저장된 디렉토리 경로
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.template_dir = template_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 템플릿 환경 설정
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_report(self, title: str) -> str:
        """보고서 생성
        
        Args:
            title: 보고서 제목
            
        Returns:
            생성된 보고서 파일 경로
        """
        # 추론 결과 로드
        inference_results = load_inference_results(self.input_dir)
        
        if not inference_results:
            print("추론 결과가 없습니다.")
            return ""
        
        # 결과 요약 생성
        summary = self._create_summary(inference_results)
        
        # 보고서 데이터 생성
        report_data = ReportData(
            title=title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            inference_results=inference_results,
            summary=summary
        )
        
        # 마크다운 보고서 생성
        markdown_content = self._generate_markdown(report_data)
        
        # 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(markdown_content)
        
        print(f"보고서가 생성되었습니다: {filepath}")
        return filepath
    
    def _create_summary(self, inference_results: List[InferenceResult]) -> Dict[str, Any]:
        """추론 결과 요약 생성
        
        Args:
            inference_results: 추론 결과 리스트
            
        Returns:
            요약 정보
        """
        frameworks = {}
        devices = {}
        
        for result in inference_results:
            # 프레임워크별 카운트
            framework = result.framework
            frameworks[framework] = frameworks.get(framework, 0) + 1
            
            # 디바이스별 카운트
            device = result.metadata.get("device", "unknown")
            devices[device] = devices.get(device, 0) + 1
        
        return {
            "total_inferences": len(inference_results),
            "frameworks": frameworks,
            "devices": devices,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _generate_markdown(self, report_data: ReportData) -> str:
        """마크다운 보고서 생성
        
        Args:
            report_data: 보고서 데이터
            
        Returns:
            마크다운 형식의 보고서 내용
        """
        # 기본 템플릿 생성
        template = """
# {{ report_data.title }}

생성 시간: {{ report_data.timestamp }}

## 요약

- 총 추론 횟수: {{ report_data.summary.total_inferences }}

### 프레임워크별 통계
{% for framework, count in report_data.summary.frameworks.items() %}
- {{ framework }}: {{ count }}개
{% endfor %}

### 디바이스별 통계
{% for device, count in report_data.summary.devices.items() %}
- {{ device }}: {{ count }}개
{% endfor %}

## 상세 추론 결과

{% for result in report_data.inference_results %}
### 추론 {{ loop.index }}

- 모델: {{ result.model_name }}
- 프레임워크: {{ result.framework }}
- 입력 데이터 형태: {{ result.input_data }}
- 출력: {{ result.output }}
- 디바이스: {{ result.metadata.device }}
- 신뢰도: {{ result.metadata.confidence }}

{% endfor %}
"""
        
        # 템플릿 렌더링
        return jinja2.Template(template).render(report_data=report_data) 