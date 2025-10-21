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
    通用 OCR 處理函數
    """
    if not prompt:
        prompt = "<image>\nFree OCR."
    
    os.makedirs("./outputs", exist_ok=True)
    
    # 清理之前的輸出文件
    for ext in ['*.mmd', '*.json', '*.txt', '*.md']:
        for f in glob.glob(f"./outputs/{ext}"):
            try:
                os.remove(f)
            except:
                pass
    
    # 驗證圖片文件存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"圖片文件不存在: {image_path}")
    
    # 調用模型
    model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path="./outputs",
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=True
    )

    # 從輸出文件讀取結果
    text = ""
    
    # 優先檢查 .mmd 文件
    mmd_files = glob.glob("./outputs/*.mmd")
    if mmd_files:
        with open(mmd_files[0], 'r', encoding='utf-8') as f:
            text = f.read().strip()
            text = text.strip('"')
    
    # 其次檢查 JSON 文件
    if not text:
        json_files = glob.glob("./outputs/*.json")
        if json_files:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get('text', '')
    
    # 最後檢查 TXT 文件
    if not text:
        txt_files = glob.glob("./outputs/*.txt")
        if txt_files:
            with open(txt_files[0], 'r', encoding='utf-8') as f:
                text = f.read()
    
    if not text:
        text = "No output file found"
    
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return text, lines

def run_ocr(image_url: str, prompt: str = None) -> Tuple[str, List[str]]:
    """
    從 URL 下載圖片並進行 OCR
    """
    tmp_path = "./outputs/tmp_input.jpg"
    _download_image(image_url, tmp_path)
    return _process_ocr(tmp_path, prompt)

def run_ocr_local(image_path: str, prompt: str = None) -> Tuple[str, List[str]]:
    """
    直接使用本地圖片進行 OCR（隱私模式）
    
    Args:
        image_path: 本地圖片路徑，例如 "F:/images/test.jpg"
        prompt: 可選的提示詞
    
    Returns:
        (full_text, lines): OCR 結果
    """
    return _process_ocr(image_path, prompt)

def runtime_meta() -> dict:
    return {
        "device": DEVICE,
        "model_id": MODEL_ID,
        "torch": torch.__version__,
    }