# TryOnAPI

가상 착용(Virtual Try-On) API 서비스입니다. 이 API는 사용자 이미지와 의류 이미지를 받아 가상 착용 결과를 생성합니다.

## 기능

1. 사용자 이미지 처리
   - MediaPipe를 사용한 키포인트 추출 (Class 17)
   - 이미지 세그멘테이션 (Class 20)
   - 배경 제거
   - Agnostic 이미지 생성

2. 의류 이미지 처리
   - 세그멘테이션
   - 준비

3. Diffusion 모델을 사용한 가상 착용
   - 256x256 기본 모델
   - 512x512 업스케일링
   - 1024x1024 슈퍼 해상도
   - 원본 비율 복원

## 설치 및 실행

### 요구 사항

- Python 3.8 이상
- 필요한 라이브러리 (requirements.txt 참조)

### 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/TryOnAPI.git
cd TryOnAPI

# 가상 환경 생성 및 활성화 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# 필요한 디렉토리 생성
mkdir -p temp output models
```

### 실행

```bash
uvicorn main:app --reload
```

서버가 시작되면 `http://localhost:8000` 에서 API에 접근할 수 있습니다.
API 문서는 `http://localhost:8000/docs` 에서 확인할 수 있습니다.

### Docker를 사용한 실행

```bash
# 도커 이미지 빌드
docker build -t tryon-api .

# 컨테이너 실행
docker run -p 8000:8000 tryon-api
```

## API 사용법

### 전체 가상 착용 프로세스

사용자 이미지와 의류 이미지를 한 번에 처리합니다.

```
POST /api/tryon/complete
```

### 단계별 처리

1. 사용자 이미지 업로드
```
POST /api/tryon/upload/user-image
```

2. 의류 이미지 업로드
```
POST /api/tryon/upload/clothing
```

3. MediaPipe 키포인트 추출
```
POST /api/mediapipe/keypoints
```

4. 이미지 세그멘테이션
```
POST /api/segmentation/process
```

5. 배경 제거
```
POST /api/segmentation/remove-background
```

6. Agnostic 이미지 생성
```
POST /api/segmentation/create-agnostic
```

7. Diffusion 모델 처리
```
POST /api/diffusion/256
POST /api/diffusion/512
POST /api/diffusion/1024
```

8. 원본 비율 복원
```
POST /api/diffusion/restore-ratio
```

## 플로우 차트

TryOnAPI.drawio.drawio.png 파일에 API 프로세스 플로우가 있습니다.

## 라이센스

[라이센스 정보]