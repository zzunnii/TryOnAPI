# TryOnAPI

가상 의류 피팅을 위한 FastAPI 기반 API 서버. 인체 파싱, 배경 제거, 세그멘테이션 등의 기능을 제공합니다.

## 주요 기능

1. **인체 파싱**: ZhengPeng7/BiRefNet_HR 모델을 사용하여 이미지에서 사람을 인식하고 배경을 제거합니다.
2. **세그멘테이션**: 인체 구성 요소(머리카락, 얼굴, 상의, 하의 등)를 세그멘테이션하여 20개 클래스로 분류합니다.
3. **JSON 출력**: 세그멘테이션 결과를 JSON 형태로 반환하여 다른 애플리케이션에서 활용할 수 있습니다.

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/username/TryOnAPI.git
cd TryOnAPI

# 필요한 패키지 설치
pip install -r requirements.txt

# BiRefNet_HR 모델 다운로드 (Hugging Face)
# huggingface-cli login  # 필요한 경우
# 로그인 후 모델 파일이 자동으로 다운로드됩니다
```

## 사용 방법

### 서버 실행

```bash
# 개발 서버 실행
python main.py

# 또는 uvicorn으로 실행
uvicorn main:app --reload
```

### API 엔드포인트

#### 인체 파싱

- **URL**: `/api/human-parsing/process`
- **메서드**: POST
- **형식**: multipart/form-data
- **파라미터**: 
  - `image`: 이미지 파일
  - `use_firebase`: Firebase 사용 여부 (기본값: false)
  - `save_visualization`: 시각화 이미지 저장 여부 (기본값: true)
- **응답**: 
  ```json
  {
    "file_id": "unique-id",
    "original_filename": "example.jpg",
    "json_path": "/output/human_parsing/example_mask.json",
    "class_count": 18,
    "image_size": {
      "width": 1080,
      "height": 1920
    },
    "visualization_path": "/output/human_parsing/example_visualization.png",
    "message": "이미지 처리 완료",
    "success": true
  }
  ```

### 테스트 스크립트 실행

```bash
python test_function.py --image path/to/image.jpg --visualize
```

## 디렉토리 구조

```
TryOnAPI/
├── app/                     # FastAPI 애플리케이션
│   ├── core/                # 핵심 설정 및 유틸리티
│   │   ├── config.py        # 애플리케이션 설정
│   ├── models/              # 모델 정의
│   │   └── parsingModels/   # 인체 파싱 모델
│   ├── routers/             # API 라우터
│   │   ├── human_parsing.py # 인체 파싱 API 라우트
│   │   └── segmentation.py  # 세그멘테이션 API 라우트
│   └── services/            # 비즈니스 로직 서비스
│       ├── human_parsing/   # 인체 파싱 서비스
│       └── segmentation_service.py # 세그멘테이션 서비스
├── models/                  # 모델 체크포인트
├── output/                  # 처리 결과 저장
├── temp/                    # 임시 파일 저장
├── utils/                   # 공통 유틸리티
│   ├── firebase_utils.py    # Firebase 연동 (주석 처리)
│   └── statistics_summary.json # BiRefNet 모델 통계
├── test_function.py         # 테스트 스크립트
├── main.py                  # FastAPI 메인 앱
└── requirements.txt         # 의존성 패키지
```

## 향후 계획

1. Firebase 연동 완료
2. 성능 최적화
3. 배치 처리 지원
4. 웹 인터페이스 추가

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.
