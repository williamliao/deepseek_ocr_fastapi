from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import shutil
from pathlib import Path
import tempfile
import os
import traceback

from app.ocr import run_ocr, run_ocr_local, runtime_meta, pdf_to_images_high_quality

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

@app.post("/ocr/pdf")
async def ocr_pdf_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form(None), 
    password: str = Form(None)
):
    """
    上傳 PDF 檔，會自動分頁轉圖片並逐頁 OCR。
    回傳整合後全文與每頁分行結果。
    """
    print("\n[DEBUG] --- /ocr/pdf endpoint hit ---")
    if password:
        print(f"[DEBUG] Received password: {'*' * len(password)}")
    try:
        # === 1. 暫存上傳的 PDF ===
        print("[DEBUG] 1. Saving PDF to temp file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        print(f"[DEBUG] 1. PDF saved to: {tmp_path}")

        # === 2. 將 PDF 拆成圖片 ===
        print("[DEBUG] 2. Converting PDF to images...")
        images = pdf_to_images_high_quality(
            tmp_path, 
            dpi=200, 
            user_password=password
        )
        print(f"[DEBUG] 2. PDF converted to {len(images)} image(s).")
        os.remove(tmp_path)
        print(f"[DEBUG] 2. Temp PDF file {tmp_path} removed.")

        all_text = ""
        all_lines = []
        page_texts = []

        # === 3. 逐頁 OCR ===
        print("[DEBUG] 3. Starting page-by-page OCR loop...")
        for idx, img in enumerate(images):
            print(f"[DEBUG] 3. Processing page {idx + 1}...")
            
            # *** 檢查 'outputs' 資料夾是否存在 ***
            output_dir = Path("./outputs")
            output_dir.mkdir(exist_ok=True) # <-- 自動建立資料夾
            
            page_tmp = str(output_dir / f"tmp_page_{idx}.jpg")
            
            print(f"[DEBUG] 3a. Saving image to {page_tmp}...")
            img.save(page_tmp) # <-- 如果 outputs 資料夾不存在，這裡會出錯
            print(f"[DEBUG] 3a. Image saved.")

            print(f"[DEBUG] 3b. Running run_ocr_local on {page_tmp}...")
            text, lines = run_ocr_local(page_tmp, prompt) # <-- 核心 OCR 步驟
            print(f"[DEBUG] 3b. OCR complete for page {idx + 1}.")
            
            all_text += f"\n\n[Page {idx+1}]\n" + text
            all_lines.append(lines)
            page_texts.append({
                "page": idx + 1,
                "text": text,
                "lines": lines
            })

            os.remove(page_tmp)
            print(f"[DEBUG] 3c. Removed temp image {page_tmp}.")

        print("[DEBUG] 4. All pages processed. Returning JSON.")
        return JSONResponse({
            "status": "ok",
            "pages": page_texts,
            "text_full": all_text.strip(),
            "meta": runtime_meta()
        })

    except Exception as e:
        # --- *** 這就是我們的新偵錯區塊 *** ---
        print("\n--- !!! AN EXCEPTION OCCURRED !!! ---")
        print(f"Error Type: {type(e)}")
        print(f"Error Details: {str(e)}")
        print("--- FULL TRACEBACK ---")
        traceback.print_exc()  # <--- 強制在 console 印出完整錯誤
        print("-------------------------\n")
        
        # 並且也將錯誤訊息回傳給 n8n
        return JSONResponse(
            {
                "status": "error", 
                "error": str(e),
                "traceback": traceback.format_exc() # 將 traceback 訊息也傳回去
            }, 
            status_code=500
        )

@app.post("/pdf/split")
async def pdf_split_endpoint(
    file: UploadFile = File(...),
    password: str = Form(None)
):
    """
    接收 PDF 檔，解密並將其拆分為多張圖片。
    回傳儲存的圖片路徑列表，供後續逐頁 OCR。
    """
    print("\n[DEBUG] --- /pdf/split endpoint hit ---")
    
    # 建立一個唯一的資料夾來存放該 PDF 的所有頁面
    # 這樣可以避免同時處理多個 PDF 時檔案名稱衝突
    output_dir = Path(f"./outputs/pdf_{int(time.time())}_{file.filename[:20]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Pages will be saved to: {output_dir.resolve()}")

    try:
        # === 1. 暫存上傳的 PDF ===
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # === 2. 將 PDF 拆成圖片 (使用您已有的 ocr.py 函式) ===
        print(f"[DEBUG] Converting PDF with password: {'*' * len(password) if password else 'None'}")
        images = pdf_to_images_high_quality(
            tmp_path, 
            dpi=200, 
            user_password=password
        )
        print(f"[DEBUG] PDF converted to {len(images)} image(s).")
        os.remove(tmp_path) # 刪除暫存 PDF

        # === 3. 儲存所有圖片並收集路徑 ===
        image_paths = []
        for idx, img in enumerate(images):
            # 儲存圖片到我們建立的唯一資料夾中
            page_path = output_dir / f"page_{idx + 1}.jpg"
            img.save(page_path)
            
            # 必須回傳「絕對路徑」，run_ocr_local 才能找到檔案
            image_paths.append(page_path.resolve().as_posix())

        print(f"[DEBUG] Returning {len(image_paths)} image paths.")
        # 立即回傳，這個請求非常快
        return {
            "status": "ok",
            "page_count": len(image_paths),
            "image_paths": image_paths # <--- 這是 n8n 需要的陣列
        }

    except Exception as e:
        print("\n--- !!! AN EXCEPTION OCCURRED in /pdf/split !!! ---")
        traceback.print_exc()
        print("-------------------------\n")
        raise HTTPException(
            status_code=500, 
            detail={"status": "error", "error": str(e)}
        )