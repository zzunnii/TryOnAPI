from pydantic import BaseModel
from typing import Optional, List

class ImageBase(BaseModel):
    filename: str
    file_size: int
    mime_type: str

class OriginalImage(ImageBase):
    width: int
    height: int
    
class SegmentedImage(ImageBase):
    original_image_id: str
    
class AgnosticImage(ImageBase):
    segmented_image_id: str
    
class DiffusionModelInput(BaseModel):
    agnostic_image_id: str
    target_clothing_id: str
    prompt: Optional[str] = None
    
class DiffusionOutput(ImageBase):
    model_name: str
    size: str  # 256x256, 512x512, 1024x1024
    input_id: str