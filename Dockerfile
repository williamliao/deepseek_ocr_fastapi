# 🚀 最新配置：CUDA 12.8 + 預編譯 Flash Attention
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# 1️⃣ 系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip wget curl ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2️⃣ 升級 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3️⃣ 安裝 PyTorch（使用 cu124，與 Flash Attention 匹配）
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 4️⃣ 【關鍵】下載並安裝預編譯的 Flash Attention
RUN wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu124torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    && pip install --no-cache-dir flash_attn-2.8.3+cu124torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    && rm flash_attn-2.8.3+cu124torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 5️⃣ 驗證所有組件
RUN python -c "\
import torch; \
import torchvision; \
import flash_attn; \
print('='*70); \
print('📦 組件版本'); \
print('='*70); \
print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ TorchVision: {torchvision.__version__}'); \
print(f'✅ Flash Attention: {flash_attn.__version__}'); \
print(f'✅ CUDA 容器版本: 12.8.1'); \
print(f'✅ PyTorch CUDA 版本: {torch.version.cuda}'); \
print('\\n' + '='*70); \
print('🔧 CUDA 狀態'); \
print('='*70); \
print(f'✅ CUDA 可用: {torch.cuda.is_available()}'); \
if torch.cuda.is_available(): \
    print(f'✅ 支援的架構: {torch.cuda.get_arch_list()}'); \
    print(f'✅ BFloat16 支援: {torch.cuda.is_bf16_supported()}'); \
else: \
    print('⚠️  CUDA 不可用（構建時正常，運行時需要 --gpus all）'); \
print('='*70); \
"

# 6️⃣ 應用依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7️⃣ 應用代碼
COPY . .

# 8️⃣ 環境變量（支援 Blackwell 及以下所有架構）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0" \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# 9️⃣ 創建緩存目錄
RUN mkdir -p /app/.cache/huggingface

# 🔟 非 root 用戶
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 1️⃣1️⃣ 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

EXPOSE 8003

# 1️⃣2️⃣ 啟動服務
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8003", \
     "--workers", "1", \
     "--timeout-keep-alive", "300"]