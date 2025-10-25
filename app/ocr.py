import io
import os
import base64
import time
import requests
import torch
import glob
import json
from pathlib import Path
from PIL import Image
from typing import Tuple, List
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
import fitz  # PyMuPDF
import io
import img2pdf
from PIL import Image
import tempfile
import fnmatch
import sys

# 初始化
load_dotenv()

MODEL_ID = os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# 載入模型
if DEVICE == "cuda":
    try:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"Flash attention failed: {e}, falling back to eager...")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )

    except Exception as e:
        print(f"Flash attention failed: {e}")
        
else:
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

model = model.eval()

# ===============================
# 輔助函式
# ===============================
def _download_image(image_url: str, save_path: str):
    r = requests.get(image_url, timeout=30)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)
    return save_path

def _process_ocr(image_path: str, prompt: str = None) -> Tuple[str, List[str]]:
    """
    DeepSeek-OCR 處理（從標準輸出捕捉結果）
    """
    from io import StringIO
    import re
    
    if not prompt:
        prompt = "<image>\nFree OCR."
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"圖片文件不存在: {image_path}")

    print(f"[DEBUG-OCR] 開始處理: {os.path.basename(image_path)}")
    
    text = ""
    output_dir = os.path.abspath("./ocr_output_debug")
    os.makedirs(output_dir, exist_ok=True)
    
    # 捕捉標準輸出和標準錯誤
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = StringIO()
    sys.stderr = captured_stderr = StringIO()
    
    try:
        # 調用模型（不打印內部信息）
        model.infer(
            tokenizer,
            prompt=prompt,
            image_file=os.path.abspath(image_path),
            output_path=output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            test_compress=False,
            eval_mode=False
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    # 獲取捕捉的輸出
    stdout_text = captured_stdout.getvalue()
    stderr_text = captured_stderr.getvalue()
    
    print(f"[DEBUG-OCR] ✓ 模型執行完成")
    
    # ====================================================================
    # 首先嘗試從檔案讀取（優先）
    # ====================================================================
    all_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    
    if all_files:
        print(f"[DEBUG-OCR] 找到 {len(all_files)} 個檔案，嘗試讀取...")
        
        priority_exts = ['.mmd', '.txt', '.md', '.json']
        for ext in priority_exts:
            matching_files = [f for f in all_files if f.lower().endswith(ext)]
            
            for file_path in matching_files:
                if any(file_path.lower().endswith(img_ext) for img_ext in ['.jpg', '.png', '.jpeg']):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if ext == '.json':
                        import json
                        try:
                            data = json.loads(content)
                            if isinstance(data, dict):
                                text = data.get('text', '') or data.get('content', '')
                            elif isinstance(data, list) and len(data) > 0:
                                text = str(data[0])
                            if not text:
                                text = json.dumps(data, ensure_ascii=False, indent=2)
                        except:
                            text = content
                    else:
                        text = content.strip().strip('"')
                    
                    if text and len(text) > 10:
                        print(f"[DEBUG-OCR] ✓ 從檔案讀取成功")
                        break
                except:
                    continue
            
            if text:
                break
    
    # ====================================================================
    # 如果檔案中沒有找到，從標準輸出提取
    # ====================================================================
    if not text and stdout_text:
        print(f"[DEBUG-OCR] 從標準輸出提取...")
        
        # 清理捕捉的輸出
        clean_text = stdout_text
        
        # 移除 ANSI 控制碼
        clean_text = re.sub(r'\x1b\[[0-9;]*m', '', clean_text)
        
        # 移除我們自己的調試信息
        debug_patterns = [
            r'\[內部\][^\n]*\n',
            r'\[DEBUG[^\]]*\][^\n]*\n',
            r'={3,}\n',
            r'BASE: torch\.Size\([^\)]+\)\n',
            r'PATCHES: torch\.Size\([^\)]+\)\n',
        ]
        
        for pattern in debug_patterns:
            clean_text = re.sub(pattern, '', clean_text, flags=re.MULTILINE)
        
        # 移除開頭和結尾的空白
        clean_text = clean_text.strip()
        
        # 如果有 Markdown 代碼塊，提取內容
        if '```' in clean_text:
            parts = clean_text.split('```')
            if len(parts) >= 3:
                # 取第一個代碼塊的內容
                text = parts[1].strip()
                # 移除可能的語言標識（如 markdown, text 等）
                if '\n' in text:
                    lines = text.split('\n')
                    if lines[0].strip() in ['markdown', 'text', 'md', '']:
                        text = '\n'.join(lines[1:])
            elif len(parts) == 2:
                text = parts[1].strip()
        else:
            text = clean_text
        
        if text:
            print(f"[DEBUG-OCR] ✓ 從標準輸出提取成功")
    
    # ====================================================================
    # 最終驗證
    # ====================================================================
    if not text or len(text) < 10:
        print(f"[DEBUG-OCR] ✗ 未能獲取有效的 OCR 結果")
        print(f"[DEBUG-OCR] 標準輸出長度: {len(stdout_text)}")
        print(f"[DEBUG-OCR] 標準錯誤長度: {len(stderr_text)}")
        raise Exception("無法從模型獲取有效的 OCR 結果")
    
    print(f"[DEBUG-OCR] ✅ 成功! 文字長度: {len(text)} 字元")
    
    # 預覽前 150 字元
    preview = text[:150].replace('\n', ' ')
    if len(text) > 150:
        preview += '...'
    print(f"[DEBUG-OCR] 預覽: {preview}")
    
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return text, lines

def run_ocr(image_url: str, prompt: str = None) -> Tuple[str, List[str]]:
    """
    從 URL 下載圖片並執行 OCR
    """
    try:
        response = requests.get(image_url, timeout=30) # 30 秒超時
        response.raise_for_status() # 如果下載失敗 (例如 404) 會拋出錯誤
        
        # 將下載的圖片寫入臨時文件
        # 我們需要一個帶有正確副檔名的文件，以便模型可以識別它
        suffix = Path(image_url).suffix or '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(response.content)
            tmp_image_path = tmp_file.name

        # 呼叫通用的本地 OCR 處理器
        full_text, lines = _process_ocr(tmp_image_path, prompt)
        
        # 清理臨時圖片
        os.remove(tmp_image_path)
        
        return full_text, lines
        
    except requests.RequestException as e:
        raise Exception(f"Failed to download image from URL: {e}")
    except Exception as e:
        # 捕捉 _process_ocr 可能拋出的錯誤
        # 確保即使出錯也刪除臨時文件
        if 'tmp_image_path' in locals() and os.path.exists(tmp_image_path):
            os.remove(tmp_image_path)
        raise e

def run_ocr_local(image_path: str, prompt: str = None) -> Tuple[str, List[str]]:
    """
    直接使用本地圖片進行 OCR（隱私模式）
    (此函式是 _process_ocr 的一個簡單包裝器)
    """
    # 確保模型已載入
    if not model or not tokenizer:
        raise Exception("AI model is not loaded. Server configuration error.")
        
    return _process_ocr(image_path, prompt)

def pdf_to_images_high_quality(
    pdf_path: str, 
    dpi: int = 144, 
    image_format: str = "PNG", 
    user_password: str = None
) -> List[Image.Image]:
    """
    將 PDF 逐頁轉成高品質 PIL 圖片列表
    (*** 修正版：支援加密 PDF ***)
    """
    images = []
    
    # 1. 打開 PDF 文件
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        # 如果檔案路徑或基本開啟失敗
        raise Exception(f"Failed to open PDF file at {pdf_path}: {e}")

    # 2. 檢查是否加密，並嘗試解密
    if pdf_document.is_encrypted:
        print(f"[DEBUG-PDF] PDF is encrypted. Attempting authentication...")
        if not user_password:
            # 如果文件已加密，但調用者沒有提供密碼
            pdf_document.close()
            raise Exception("Document is encrypted, but no password was provided.")
        
        # 嘗試使用密碼解密
        auth_success = pdf_document.authenticate(user_password)
        if not auth_success:
            # 密碼錯誤
            pdf_document.close()
            raise Exception("Invalid password provided for PDF.")
        
        print(f"[DEBUG-PDF] PDF authenticated successfully.")
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    page_num = 0

    try:
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            Image.MAX_IMAGE_PIXELS = None # 移除 PIL 的圖片大小限制

            img_data = pixmap.tobytes("png") # 始終使用 png 獲取原始數據
            img = Image.open(io.BytesIO(img_data))

            # 若為 RGBA 轉成 RGB (JPEGs 不支援透明度)
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background

            images.append(img)
            
    except Exception as e:
        # 捕捉轉檔過程中的錯誤 (例如 "document closed or encrypted")
        pdf_document.close()
        raise Exception(f"Error during PDF page conversion (page {page_num}): {e}")

    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images, output_path: str):
    """
    將多張 PIL 圖片轉成單一 PDF

    Args:
        pil_images (List[Image.Image]): 圖片列表
        output_path (str): 轉換後的 PDF 輸出路徑
    """
    if not pil_images:
        raise ValueError("沒有圖片可轉換")

    image_bytes_list = []
    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        image_bytes_list.append(img_buffer.getvalue())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pdf_bytes = img2pdf.convert(image_bytes_list)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)

def runtime_meta() -> dict:
    return {
        "device": DEVICE,
        "model_id": MODEL_ID,
        "torch": torch.__version__,
    }