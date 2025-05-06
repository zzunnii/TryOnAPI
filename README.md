# TryOnAPI

TryOnAPI는 최첨단 딥러닝 모델을 활용하여 사실적인 의류 가상 피팅을 가능하게 하는 종합적인 API 서비스입니다.

## Features

- IDM-VTON 모델을 활용한 가상 피팅
- 인체 파싱 및 세그멘테이션
- DensePose 분석
- 포즈 추정 (OpenPose)
- 다양한 의류 카테고리에 대한 마스크 생성
- 이미지 처리 유틸리티
- RESTful API 인터페이스

## System Requirements

- Python 3.8 이상
- CUDA 호환 GPU (권장)
- 16GB 이상의 VRAM
- RTX 4060ti 16GB 기준 평균 12분 소요

## Installation

### Setup Environment

```bash
# Conda 환경 생성
conda create -n tryonapi python=3.8
conda activate tryonapi

# 요구사항 설치

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### Model Setup

필요한 모델 체크포인트 다운로드:

```bash
# 모델 설정 스크립트 실행
python setup_model.py
```

## Project Structure

```
api/
├── core/                # 핵심 설정
├── densepose/           # DensePose 모듈
├── diffusionModels/     # IDM-VTON 확산 모델
├── models/              # 모델 체크포인트 디렉토리
├── output/              # 생성된 이미지의 출력 디렉토리
├── preprocess/          # 전처리 모듈
│   ├── humanparsing/    # 인체 파싱 모델
│   └── openpose/        # OpenPose 구현
├── routers/             # API 엔드포인트용 FastAPI 라우터
├── services/            # 모델 서비스
│   ├── densepose_service.py
│   └── model_service.py
├── utils/               # 유틸리티 함수
│   └── image_utils.py   # 이미지 처리 유틸리티
└── main.py              # 메인 애플리케이션 진입점
```

## Usage

### Starting the API Server

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation

서버가 실행되면 대화형 API 문서에 접근할 수 있습니다:

```
http://localhost:8000/docs
```

## API Endpoints

### Preprocessing Endpoints

- `POST /api/preprocess/human-parsing`: 업로드된 이미지에 대해 인체 파싱 수행
- `POST /api/preprocess/densepose`: 이미지에 DensePose 분석 적용
- `POST /api/preprocess/pose-estimation`: 이미지에서 포즈 키포인트 추출
- `POST /api/preprocess/mask-generation`: 의류 카테고리 마스크 생성

### Virtual Try-On Endpoints

- `POST /api/tryon`: 가상 피팅을 위한 주요 엔드포인트
- `GET /api/tryon/{tryon_id}`: 특정 피팅 결과 조회

## Implementation Details

### Image Processing Pipeline

1. **인체 파싱**: 인체를 다양한 의미론적 부분으로 분할
2. **포즈 추정**: OpenPose를 사용하여 골격 키포인트 추출
3. **DensePose**: 2D 이미지 픽셀을 3D 신체 표면에 매핑
4. **마스크 생성**: 대상 의류 영역에 대한 마스크 생성
5. **가상 피팅**: 사실적인 의류 전송을 위한 IDM-VTON 확산 모델 적용

### Model Architecture

시스템은 주로 Stable Diffusion XL을 기반으로 한 IDM-VTON 모델을 사용하며, 다음과 같은 커스텀 컴포넌트를 포함합니다:

- 의류 특징 추출을 위한 수정된 UNet 아키텍처
- 의류 정렬을 위한 특수 어텐션 프로세서
- 잠재 공간 압축 및 재구성을 위한 VAE

## Development

### Adding New Features

새 기능을 추가하려면:

1. 적절한 모듈에 기능 구현
2. 라우터에 필요한 API 엔드포인트 추가
3. 문서 업데이트

### Testing

다음과 같이 테스트 실행:

```bash
pytest tests/
```

## License

이 프로젝트는 CC BY-NC-SA (Creative Commons Attribution-NonCommercial-ShareAlike) 라이선스를 따릅니다.

- **저작자 표시(BY)**: 원작자를 적절히 표시해야 합니다.
- **비영리(NC)**: 상업적 목적으로 사용할 수 없습니다.
- **동일조건변경허락(SA)**: 이 라이선스와 동일한 조건으로 2차 저작물을 배포해야 합니다.

## Original Author

이 프로젝트는 다음 원본 프로젝트를 기반으로 합니다:
- **원작자**: yisol
- **원본 프로젝트**: [IDM-VTON](https://github.com/yisol/IDM-VTON)

## Acknowledgements

- 가상 피팅을 위한 IDM-VTON 모델
- 포즈 추정을 위한 OpenPose
- 밀도 높은 인체 포즈 추정을 위한 DensePose
- 세그멘테이션을 위한 인체 파싱 모델

## Contact

[연락처 정보 미제공]
