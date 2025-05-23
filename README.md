# IDM-VTON FastAPI

IDM-VTON을 FastAPI로 구현한 가상 피팅 API 서비스입니다.

## Test
## 🖼️ 가상 피팅 결과 예시

| 원본 이미지 (Person)               | 착용 의류 (Clothing) | 결과 이미지 (Try-On Result) |
|-------------------------------|-----------------------|-------------------------------|
| ![]() | ![](./assets/cloth.jpg) | ![](./assets/result.jpg) |

## 환경 요구사항

- Windows 10/11
- Anaconda
- CUDA 지원 GPU (권장)
- Python 3.8 이상

## 설치 방법

### 1. Detectron2 환경 설정

```bash
# Detectron2를 위한 새로운 conda 환경 생성
conda create -n detectron python=3.8
conda activate detectron

# PyTorch 설치
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# Detectron2 의존성 설치
pip install fvcore
pip install ninja
pip install cython
pip install cloudpickle
pip install omegaconf
pip install pycocotools
pip install opencv-python
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
pip install uvicorn
pip install uv
pip install timm
```

### 2. TryOn API 환경 설정

```bash
# TryOn API를 위한 새로운 conda 환경 생성
conda create -n tryon python=3.8
conda activate tryon

# FastAPI 및 기타 의존성 설치
pip install -r requirements.txt
```

## 설정

1. `core/config.py` 파일에서 다음 설정을 수정하세요:
   ```python
   DENSEPOSE_API_URL: str = "your_densepose_api_url_here"
   ```

2. `test_api.py` 파일에서 API URL을 적절히 수정하세요.

## 모델 설정

1. Detectron2 환경에서:
```bash
conda activate detectron
# Detectron2 관련 설정 실행
```

2. TryOn API 환경에서:
```bash
conda activate tryon
python setup_model.py
```

## API 실행

TryOn API 환경에서:
```bash
conda activate tryon
uvicorn api.main:app --reload

conda activate detectron
uvicorn api.main:app --reload

python api/test_api.py --api-url <API> --person <person image> --clothing <clothe image> --category <clothe category>
```

## API 엔드포인트

- `POST /try-on`: 가상 피팅 API 엔드포인트
  - 입력: 사용자 이미지와 의류 이미지
  - 출력: 가상 피팅 결과 이미지

## 주의사항

- CUDA 지원 GPU가 있는 환경에서 실행하는 것을 권장합니다.
- Detectron2와 TryOn API는 별도의 Anaconda 환경에서 실행해야 합니다.
- 각 환경에서 필요한 의존성이 올바르게 설치되었는지 확인하세요.
- API URL 설정을 반드시 확인하세요. 

## Citation
### IDM-VTON
```
@article{choi2024improving,
  title={Improving Diffusion Models for Authentic Virtual Try-on in the Wild},
  author={Choi, Yisol and Kwak, Sangkyung and Lee, Kyungmin and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2403.05139},
  year={2024}
}
```
### Detectron2
```
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

## License
The codes and checkpoints in this repository are under the CC BY-NC-SA 4.0 license.