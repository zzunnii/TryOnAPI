"""
IDM-VTON 모델 설정 스크립트
필요한 디렉토리 생성 및 모델 다운로드
"""

import os
import sys
import shutil
from pathlib import Path

def create_required_directories():
    """필요한 디렉토리 생성"""
    # 기본 경로
    base_dir = Path(__file__).parent.absolute()
    
    # 필요한 디렉토리 목록
    required_dirs = [
        base_dir / "models" / "idm_vton",
        base_dir / "temp" / "user_images",
        base_dir / "temp" / "target_clothing",
        base_dir / "output" / "idm_vton",
        base_dir / "output" / "human_parsing",
        base_dir / "output" / "mediapipe",
        base_dir / "output" / "masks",
        base_dir / "result"
    ]
    
    # 각 디렉토리 생성
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"디렉토리 생성 완료: {dir_path}")
    
    return True

def download_idm_vton_model():
    """HuggingFace에서 IDM-VTON 모델 다운로드"""
    try:
        from huggingface_hub import snapshot_download
        import os
        import shutil
        
        # 기본 경로
        base_dir = Path(__file__).parent.absolute()
        model_dir = base_dir / "models" / "idm_vton"
        idm_dir = model_dir / "yisol_idm_vton"
        
        # 기존 모델 디렉토리 완전히 삭제
        print(f"기존 모델 디렉토리 삭제: {model_dir}")
        if os.path.exists(model_dir):
            try:
                shutil.rmtree(model_dir)
                print(f"기존 모델 디렉토리 삭제 완료")
            except Exception as e:
                print(f"디렉토리 삭제 중 오류: {e}")
                # Windows에서는 간혹 rmtree가 실패할 수 있음
                print("삭제 실패 시 수동으로 삭제 시도...")
                try:
                    if os.name == 'nt':  # Windows
                        os.system(f'rd /s /q "{model_dir}"')
                    else:
                        os.system(f'rm -rf "{model_dir}"')
                except Exception as e2:
                    print(f"수동 삭제 중 오류: {e2}")
        
        # 디렉토리 생성
        os.makedirs(idm_dir, exist_ok=True)
        print(f"모델 디렉토리 생성 완료: {idm_dir}")
        
        # 대안 1: 직접 pipeline 객체 다운로드
        print("Diffusers pipeline을 통해 모델 다운로드 시도...")
        try:
            from diffusers import DiffusionPipeline
            
            # 임시로 모델을 로드 (파일 다운로드를 위해)
            print("DiffusionPipeline 임시 로드 중...")
            pipe = DiffusionPipeline.from_pretrained(
                "yisol/IDM-VTON",
                # use_safetensors 옵션 제거 (bin 파일 사용)
                cache_dir=str(idm_dir)
            )
            print("모델 다운로드 완료!")
            del pipe  # 메모리 해제
            return True
        except Exception as e:
            print(f"Pipeline 다운로드 중 오류: {str(e)}")
            print("대안 방법으로 시도...")
        
        # 대안 2: huggingface_hub 사용 시도
        print("huggingface_hub를 통해 모델 다운로드 시도...")
        try:
            # 최신 huggingface_hub 버전용
            model_path = snapshot_download(
                repo_id="yisol/IDM-VTON",
                local_dir=str(idm_dir),
                local_dir_use_symlinks=False
            )
            print(f"모델 다운로드 완료: {model_path}")
            return True
        except Exception as e:
            print(f"snapshot_download 오류: {str(e)}")
            print("다운로드 실패")
            return False
        
        return True
        
    except ImportError:
        print("huggingface_hub 패키지가 설치되어 있지 않습니다.")
        print("pip install huggingface_hub 명령어로 설치하세요.")
        return False
    
    except Exception as e:
        print(f"모델 다운로드 중 오류 발생: {str(e)}")
        return False

def main():
    """메인 함수"""
    print("=== IDM-VTON 모델 설정 시작 ===")
    
    # 필요한 디렉토리 생성
    create_required_directories()
    
    # 모델 다운로드
    if download_idm_vton_model():
        print("모델 설정 완료!")
    else:
        print("경고: 모델 다운로드에 실패했습니다. IDM-VTON 시뮬레이션 모드로 실행됩니다.")
    
    print("=== 설정 완료 ===")

if __name__ == "__main__":
    main()