from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import time
import shutil
from pathlib import Path

from app.ocr import run_ocr, run_ocr_local, runtime_meta

app = FastAPI(title="DeepSeek OCR API")

# 原有的 URL 請求模型
class OCRRequest(BaseModel):
    image_url: str
    prompt: str = None

# 新增：本地路徑請求模型
class OCRLocalRequest(BaseModel):
    image_path: str
    prompt: str = None

@app.get("/")
def read_root():
    return {"message": "DeepSeek OCR FastAPI", "meta": runtime_meta()}

# 原有的 URL 端點
@app.post("/ocr")
def ocr_endpoint(req: OCRRequest):
    try:
        t0 = time.time()
        full_text, lines = run_ocr(req.image_url, req.prompt)
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "full_text": full_text,
            "lines": lines,
            "elapsed_ms": elapsed_ms,
            "meta": runtime_meta(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增：本地文件路徑端點
@app.post("/ocr/local")
def ocr_local_endpoint(req: OCRLocalRequest):
    """
    使用本地圖片路徑進行 OCR
    Example: {"image_path": "F:/images/test.jpg"}
    """
    try:
        t0 = time.time()
        full_text, lines = run_ocr_local(req.image_path, req.prompt)
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "full_text": full_text,
            "lines": lines,
            "elapsed_ms": elapsed_ms,
            "meta": runtime_meta(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增：文件上傳端點
@app.post("/ocr/upload")
async def ocr_upload_endpoint(
    file: UploadFile = File(...),
    prompt: str = None
):
    """
    上傳圖片文件進行 OCR
    支持的格式: jpg, jpeg, png, bmp, gif
    """
    try:
        # 檢查文件類型
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式。支持: {', '.join(allowed_extensions)}"
            )
        
        # 保存上傳的文件
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        temp_file = upload_dir / f"upload_{int(time.time())}{file_ext}"
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 執行 OCR
        t0 = time.time()
        full_text, lines = run_ocr_local(str(temp_file), prompt)
        elapsed_ms = int((time.time() - t0) * 1000)
        
        # 清理臨時文件（可選）
        # temp_file.unlink()
        
        return {
            "full_text": full_text,
            "lines": lines,
            "elapsed_ms": elapsed_ms,
            "meta": runtime_meta(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))