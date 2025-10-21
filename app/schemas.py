from pydantic import BaseModel, Field
from typing import List, Optional, Any

class OCRRequest(BaseModel):
    image_url: Optional[str] = Field(default=None, description="HTTP(S) 圖片連結")
    image_base64: Optional[str] = Field(default=None, description="不含 data: 前綴的 Base64 圖片資料")

class OCRResponse(BaseModel):
    full_text: str
    lines: List[str]
    elapsed_ms: int
    meta: dict

class HealthResponse(BaseModel):
    status: str
    device: str
    model_id: str
    backend: str
    extra: Optional[Any] = None
