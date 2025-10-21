# DeepSeek OCR FastAPI 服務

基於 DeepSeek OCR 模型的 FastAPI 服務，支持圖片文字識別、文檔轉 Markdown、圖表解析等功能。提供文件上傳、本地路徑和 URL 三種調用方式，支持 GPU 加速和 Docker 部署。

## 主要特性

- 🚀 **多種輸入方式**：支持文件上傳、本地路徑、URL 三種圖片輸入方式
- 📝 **靈活輸出格式**：可輸出純文本、Markdown、結構化 JSON 等多種格式
- 🎯 **多場景應用**：文檔識別、圖表解析、內容定位、圖像描述
- ⚡ **GPU 加速**：支持 CUDA 加速，提升識別速度
- 🐳 **容器化部署**：提供 Docker 鏡像，一鍵部署

---

## 環境設置

### 方式一：本機運行

#### 1. 創建並啟用虛擬環境

**使用 venv：**
```bash
# 創建虛擬環境
python -m venv .venv

# 啟用虛擬環境
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac
```

**使用 Conda：**
```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

#### 2. 安裝依賴
```bash
pip install -r requirements.txt
```

#### 3. 設置環境變量
```bash
export DEEPSEEK_OCR_MODEL="deepseek-ai/DeepSeek-OCR"
```

#### 4. 啟動服務
```bash
uvicorn app.main:app --reload --port 8000
```

### 方式二：Docker 運行（支援 GPU）
```bash
# 構建映像
docker build -t deepseek-ocr-api .

# 運行容器
docker run --gpus all \
  -p 8003:8003 \
  -e DEEPSEEK_OCR_MODEL=deepseek-ai/DeepSeek-OCR \
  --name ocr \
  deepseek-ocr-api
```

---

## API 使用方式

### 1. 文件上傳方式（推薦）

#### 使用 cURL

**Windows:**
```bash
curl -X POST "http://localhost:8000/ocr/upload" ^
  -F "file=@F:\git\deepseek_ocr_fastapi\show1.jpg" ^
  -F "prompt=Free OCR"
```

**Linux/Mac:**
```bash
curl -X POST "http://localhost:8000/ocr/upload" \
  -F "file=@/path/to/image.jpg" \
  -F "prompt=Free OCR"
```

#### 使用 Python
```python
import requests

with open("test.jpg", "rb") as f:
    files = {"file": f}
    data = {"prompt": "Free OCR"}  # 可選
    response = requests.post("http://localhost:8000/ocr/upload", files=files, data=data)
    print(response.json())
```

---

### 2. 本地路徑方式

#### 使用 cURL
```bash
curl -X POST "http://localhost:8000/ocr/local" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "F:/images/test.jpg"}'
```

#### 使用 Python
```python
import requests

response = requests.post(
    "http://localhost:8000/ocr/local",
    json={"image_path": "F:/images/test.jpg"}
)
print(response.json())
```

---

### 3. URL 方式

#### 使用 cURL

**Windows:**
```bash
curl -X POST http://localhost:8000/ocr ^
  -H "Content-Type: application/json" ^
  -d "{\"image_url\":\"https://example.com/image.jpg\"}"
```

**Linux/Mac:**
```bash
curl -X POST http://localhost:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/image.jpg"}'
```

#### 使用 Python
```python
import requests

response = requests.post(
    "http://localhost:8000/ocr",
    json={"image_url": "https://example.com/image.jpg"}
)
print(response.json())
```

---

## Prompt 配置

通過修改 `prompt` 參數來控制 OCR 輸出格式：

| 使用場景 | Prompt 示例 |
|---------|------------|
| **Markdown 格式** | `<image>\n<\|grounding\|>Convert the document to markdown.` |
| **純文本** | `<image>\nFree OCR.` |
| **基礎 OCR** | `<image>\n<\|grounding\|>OCR this image.` |
| **圖表解析** | `<image>\nParse the figure.` |
| **詳細描述** | `<image>\nDescribe this image in detail.` |
| **定位特定內容** | `<image>\nLocate <\|ref\|>xxxx<\|/ref\|> in the image.` |

---

## 示例

### 例子 1: 發票 OCR（Markdown 格式）
```python
import requests

with open("invoice.jpg", "rb") as f:
    files = {"file": f}
    data = {"prompt": "\nConvert the document to markdown."}
    response = requests.post("http://localhost:8000/ocr/upload", files=files, data=data)
    print(response.json())
```

### 例子 2: 一般文本識別
```python
import requests

response = requests.post(
    "http://localhost:8000/ocr",
    json={
        "image_url": "https://raw.githubusercontent.com/napthedev/ocr-sample/main/invoice.jpg",
        "prompt": "Free OCR"
    }
)
print(response.json())
```

---

## 注意事項

- ✅ 推薦使用**文件上傳方式**以獲得最佳性能
- 📁 本地路徑方式僅適用於服務器可訪問的路徑
- 🖥️ GPU 加速需要正確配置 NVIDIA Docker 運行時
- 🎛️ 不同的 prompt 會影響輸出格式和識別精度

---

## 技術棧

- **深度學習模型**: DeepSeek-OCR (`deepseek-ai/DeepSeek-OCR`)
- **Web 框架**: FastAPI + Uvicorn
- **Python 版本**: 3.8+
- **容器化**: Docker
- **硬件加速**: NVIDIA CUDA (可選)

---