"""
TryOnAPI 메인 애플리케이션
"""

import uvicorn
import os

from app import create_app

# FastAPI 앱 생성
app = create_app()

if __name__ == "__main__":
    # 환경 변수에서 포트 가져오기, 기본값은 8000
    port = int(os.environ.get("PORT", 8000))
    
    # 개발 모드에서 실행
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
