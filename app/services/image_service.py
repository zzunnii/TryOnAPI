import os
import aiofiles
from fastapi import UploadFile
import uuid
from PIL import Image
import io

from app.core.config import get_settings

settings = get_settings()

async def save_upload_file(uploadfile: UploadFile, filename: str, subfolder: str = "") -> str:
    """
    업로드된 파일을 저장하고 저장 경로를 반환
    """
    # 저장 디렉토리 생성
    save_dir = os.path.join(settings.TEMP_DIR, subfolder)
    os.makedirs(save_dir, exist_ok=True)
    
    # 파일 저장 경로
    file_path = os.path.join(save_dir, filename)
    
    # 파일 저장
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await uploadfile.read()
        await out_file.write(content)
    
    return file_path

def resize_image(image_path: str, max_size: int = None) -> str:
    """
    이미지 크기 조정
    """
    if max_size is None:
        max_size = settings.MAX_IMAGE_SIZE
        
    # 이미지 열기
    img = Image.open(image_path)
    
    # 원본 크기 확인
    width, height = img.size
    
    # 리사이징 필요한지 확인
    if width <= max_size and height <= max_size:
        return image_path
    
    # 비율 유지하면서 리사이징
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # 리사이징
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # 새 파일명 생성
    file_name, file_ext = os.path.splitext(image_path)
    resized_path = f"{file_name}_resized{file_ext}"
    
    # 저장
    img.save(resized_path)
    
    return resized_path