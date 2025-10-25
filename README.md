# DeepSeek OCR FastAPI 服務

基於 DeepSeek OCR 模型的 FastAPI 服務，支持圖片文字識別、文檔轉 Markdown、圖表解析等功能。提供文件上傳、本地路徑和 URL 三種調用方式，支持 GPU 加速和 Docker 部署。

## 📋 目錄

- 主要特性
- 環境要求
- 快速開始
- 環境設置
  - 方式一：本機運行
- API 使用方式
- Prompt 配置
- 示例代碼

---

## 🎯 主要特性

- 🚀 **多種輸入方式**：支持文件上傳、本地路徑、URL 三種圖片輸入方式
- 📝 **靈活輸出格式**：可輸出純文本、Markdown、結構化 JSON 等多種格式
- 🎯 **多場景應用**：文檔識別、圖表解析、內容定位、圖像描述
- ⚡ **GPU 加速**：支持 NVIDIA CUDA 加速，提升識別速度
- 🐳 **容器化部署**：提供 Docker 鏡像和 Docker Compose 配置
- 🔧 **易於集成**：RESTful API 設計，簡單易用
- 📊 **健康檢查**：內建健康檢查端點
- 🔒 **安全運行**：非 root 用戶運行

---

## 💻 環境要求

### 基礎要求
- **Python**: 3.10+
- **內存**: 建議 8GB+
- **磁盤**: 建議 20GB+（用於模型存儲）

### GPU 加速（可選）
- **NVIDIA GPU**: 計算能力 8.0+ (Ampere 架構或更新)
- **CUDA**: 12.4+
- **顯存**: 建議 8GB+

## ⚙️ 環境設置

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
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 3. 設置環境變量

```bash
# Linux/Mac
export DEEPSEEK_OCR_MODEL="deepseek-ai/DeepSeek-OCR"

# Windows PowerShell
$env:DEEPSEEK_OCR_MODEL="deepseek-ai/DeepSeek-OCR"
```

#### 4. 啟動服務

```bash
uvicorn app.main:app --reload --port 8003
```

## 📡 API 使用方式

### 1. 文件上傳方式（推薦）

最穩定可靠的方式，適合大多數場景。

#### 使用 cURL

**Windows:**
```bash
curl -X POST "http://localhost:8003/ocr/upload" ^
  -F "file=@C:\path\to\image.jpg" ^
  -F "prompt=Free OCR"
```

**Linux/Mac:**
```bash
curl -X POST "http://localhost:8003/ocr/upload" \
  -F "file=@/path/to/image.jpg" \
  -F "prompt=Free OCR"
```

#### 使用 Python

```python
import requests

# 基礎用法
with open("test.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8003/ocr/upload", files=files)
    print(response.json())

# 帶 Prompt
with open("test.jpg", "rb") as f:
    files = {"file": f}
    data = {"prompt": "<image>\n<|grounding|>Convert the document to markdown."}
    response = requests.post("http://localhost:8003/ocr/upload", files=files, data=data)
    print(response.json())
```

#### 使用 JavaScript/TypeScript

```javascript
// 使用 FormData
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('prompt', 'Free OCR');

const response = await fetch('http://localhost:8003/ocr/upload', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

---

### 2. 本地路徑方式

適用於服務器上已存在的圖片文件。

#### 使用 cURL

```bash
curl -X POST "http://localhost:8003/ocr/local" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/image.jpg", "prompt": "Free OCR"}'
```

#### 使用 Python

```python
import requests

response = requests.post(
    "http://localhost:8003/ocr/local",
    json={
        "image_path": "/path/to/image.jpg",
        "prompt": "Free OCR"
    }
)
print(response.json())
```

**注意事項：**
- 路徑必須是容器內可訪問的路徑
- 如果使用 Docker，需要先掛載目錄：`-v /host/path:/container/path`

---

### 3. URL 方式

適用於網絡上的圖片。

#### 使用 cURL

**Windows:**
```bash
curl -X POST http://localhost:8003/ocr ^
  -H "Content-Type: application/json" ^
  -d "{\"image_url\":\"https://example.com/image.jpg\", \"prompt\":\"Free OCR\"}"
```

**Linux/Mac:**
```bash
curl -X POST http://localhost:8003/ocr \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/image.jpg", "prompt":"Free OCR"}'
```

#### 使用 Python

```python
import requests

response = requests.post(
    "http://localhost:8003/ocr",
    json={
        "image_url": "https://example.com/image.jpg",
        "prompt": "Free OCR"
    }
)
print(response.json())
```

---

## 🎨 Prompt 配置

通過修改 `prompt` 參數來控制 OCR 輸出格式和行為：

| 使用場景 | Prompt 示例 | 說明 |
|---------|------------|------|
| **Markdown 格式** | `<image>\n<\|grounding\|>Convert the document to markdown.` | 將文檔轉換為 Markdown 格式 |
| **純文本** | `<image>\nFree OCR.` | 提取純文本內容 |
| **基礎 OCR** | `<image>\n<\|grounding\|>OCR this image.` | 標準 OCR 識別 |
| **圖表解析** | `<image>\nParse the figure.` | 解析圖表和數據 |
| **詳細描述** | `<image>\nDescribe this image in detail.` | 獲取圖像的詳細描述 |
| **定位特定內容** | `<image>\nLocate <\|ref\|>xxxx<\|/ref\|> in the image.` | 定位並提取特定內容 |

### Prompt 最佳實踐

1. **Markdown 轉換**：適合文檔、報告、文章等結構化內容
2. **純文本提取**：適合簡單的文字識別需求
3. **圖表解析**：適合數據圖表、統計圖等
4. **詳細描述**：適合需要理解圖像內容的場景

---

## 💡 示例代碼

### 例子 1: 發票 OCR（Markdown 格式）

```python
import requests

with open("invoice.jpg", "rb") as f:
    files = {"file": f}
    data = {"prompt": "<image>\n<|grounding|>Convert the document to markdown."}
    response = requests.post("http://localhost:8003/ocr/upload", files=files, data=data)
    
    result = response.json()
    print(result["text"])  # 印出識別的文本
```

### 例子 2: 批量處理圖片

```python
import requests
import os

def ocr_image(image_path, prompt="Free OCR"):
    """OCR 單張圖片"""
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"prompt": prompt}
        response = requests.post("http://localhost:8003/ocr/upload", files=files, data=data)
        return response.json()

# 批量處理
image_dir = "images"
results = {}

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        print(f"Processing {filename}...")
        result = ocr_image(image_path)
        results[filename] = result

# 保存結果
import json
with open("ocr_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### 例子 3: 網絡圖片識別

```python
import requests

# URL 方式
response = requests.post(
    "http://localhost:8003/ocr",
    json={
        "image_url": "https://raw.githubusercontent.com/napthedev/ocr-sample/main/invoice.jpg",
        "prompt": "Free OCR"
    }
)

result = response.json()
print(result["text"])
```

### 例子 4: 使用不同 Prompt

```python
import requests

def ocr_with_prompt(image_path, prompt_type="text"):
    """根據不同類型使用不同的 Prompt"""
    prompts = {
        "text": "<image>\nFree OCR.",
        "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "figure": "<image>\nParse the figure.",
        "detail": "<image>\nDescribe this image in detail."
    }
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"prompt": prompts[prompt_type]}
        response = requests.post("http://localhost:8003/ocr/upload", files=files, data=data)
        return response.json()

# 使用示例
result = ocr_with_prompt("document.jpg", "markdown")
print(result["text"])
```

## 🛠️ 技術棧

- **深度學習模型**: DeepSeek-OCR (`deepseek-ai/DeepSeek-OCR`)
- **Web 框架**: FastAPI 0.115.0
- **ASGI 服務器**: Uvicorn 0.30.6
- **深度學習**: PyTorch 2.5.1 + Transformers 4.46.3
- **加速**: Flash Attention 2.8.3
- **Python 版本**: 3.10
- **CUDA 版本**: 12.8.1
- **容器化**: Docker + Docker Compose

---

## 📚 相關資源

- [DeepSeek OCR 官方文檔](https://github.com/deepseek-ai/DeepSeek-OCR)
- [FastAPI 文檔](https://fastapi.tiangolo.com/)
- [Docker 文檔](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

---