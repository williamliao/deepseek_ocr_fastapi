# 1) 本機
python -m venv .venv && source .venv\Scripts\activate
pip install -r requirements.txt
export DEEPSEEK_OCR_MODEL="deepseek-ai/DeepSeek-OCR"
uvicorn app.main:app --reload --port 8000

# 2) Docker（GPU）
docker build -t deepseek-ocr-api .
docker run --gpus all -p 8003:8003 -e DEEPSEEK_OCR_MODEL=deepseek-ai/DeepSeek-OCR --name ocr deepseek-ocr-api


conda activate deepseek-ocr


curl -X POST http://localhost:8000/ocr ^
  -H "Content-Type: application/json" ^
  -d "{\"image_url\":\"https://imgv3.fotor.com/images/blog-cover-image/How-to-Make-Text-Stand-Out-And-More-Readable.jpg\"}"


用 multipart/form-data

curl -X POST http://localhost:8003/ocr ^
  -H "Content-Type: multipart/form-data" ^
  -F "body={\"image_url\":\"https://raw.githubusercontent.com/napthedev/ocr-sample/main/invoice.jpg\"};type=application/json"




使用方式：
1. 文件上傳方式（推薦）：
bash
curl -X POST "http://localhost:8000/ocr/upload" ^
  -F "file=@F:\git\deepseek_ocr_fastapi\show1.jpg" ^
  -F "prompt=Free OCR"

或使用 Python：
pythonimport requests

with open("test.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/ocr/upload", files=files)
    print(response.json())

2. 本地路徑方式：
bash
curl -X POST "http://localhost:8000/ocr/local" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "F:/images/test.jpg"}'

或使用 Python：
pythonimport requests

response = requests.post(
    "http://localhost:8000/ocr/local",
    json={"image_path": "F:/images/test.jpg"}
)
print(response.json())

3. 原有的 URL 方式仍可用：
bash
curl -X POST "http://localhost:8000/ocr" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'

# 修改 prompt 來控制輸出格式
prompt = "<image>\n<|grounding|>Convert the document to markdown."  # Markdown
# 或
prompt = "<image>\nFree OCR."  # 純文本
# other image: <image>\n<|grounding|>OCR this image.

# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.