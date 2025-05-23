"""
Detectron2 DensePose API 서버 실행 스크립트
"""

import uvicorn
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detectron2 DensePose API 서버")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="서버 호스트 주소 (기본값: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=29246,
        help="서버 포트 (기본값: 8001)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="코드 변경 시 자동 리로드 여부"
    )
    
    args = parser.parse_args()
    
    # API 서버 실행
    print(f"Detectron2 DensePose API 서버 시작 중... (http://{args.host}:{args.port})")
    uvicorn.run(
        "api:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload
    )
