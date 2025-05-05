# IDM-VTON 디버깅 가이드

이 가이드는 TryOnAPI에서 IDM-VTON 가상 착용 기능 디버깅을 위한 절차를 설명합니다.

## 필요한 디렉토리 구조 설정

먼저 필요한 디렉토리와 모델 파일을 설정하려면 다음 명령어를 실행하세요:

```bash
# 필요한 파이썬 패키지 설치
pip install huggingface_hub diffusers

# 디렉토리 구조 및 모델 설정
python setup_model.py
```

## 디버깅 절차

1. 서버 실행:
   ```bash
   # FastAPI 서버 실행
   python main.py
   ```

2. 가상 착용 테스트 실행:
   ```bash
   # 테스트 함수 실행 (기본 파라미터 사용)
   python test_function.py
   
   # 또는 직접 파라미터 지정
   python test_function.py --person "사용자이미지경로" --clothing "의류이미지경로" --category upper --output "./result/output.jpg"
   ```

3. 문제 해결 순서:
   - `setup_model.py`가 올바르게 실행되었는지 확인
   - `models/idm_vton` 디렉토리에 모델 파일이 있는지 확인
   - `temp`, `result`, `output` 등 필요한 디렉토리가 모두 생성되었는지 확인
   - 서버가 제대로 실행되고 있는지 확인 (8003번 포트)
   - 테스트 함수 출력 로그를 확인하며 어떤 단계에서 오류가 발생하는지 확인

## 비상 대비책

만약 실제 IDM-VTON 모델을 로드하는 데 계속 문제가 있다면:

1. 시뮬레이션 모드 강제 사용:
   - `idm_vton_service.py` 파일에서 `_load_model` 함수를 수정하여 항상 "simulation_mode"를 반환하도록 변경

2. 인체 파싱 건너뛰기:
   ```bash
   python test_function.py --skip-parsing
   ```

3. 마스크 생성 모드만 테스트:
   ```bash
   python test_function.py --mask-only
   ```

## 주요 디버깅 포인트

코드에 이미 다양한 디버깅 로그가 추가되었습니다. 특히 다음 부분을 주의 깊게 살펴보세요:

1. **디렉토리 구조 확인**: 코드 초기에 모든 필요한 디렉토리가 존재하는지 확인하고 누락된 경우 자동으로 생성
2. **모델 로드 과정**: IDM-VTON 모델을 로드하는 과정에서 발생하는 문제 확인
3. **이미지 처리 과정**: 이미지 로드, 전처리, 마스크 생성, 워핑 과정에서의 문제 확인
4. **파일 경로 문제**: Windows 경로(백슬래시)와 Linux 경로(슬래시) 간의 호환성 문제 확인
