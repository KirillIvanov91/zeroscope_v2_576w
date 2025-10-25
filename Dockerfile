# ====================================================
#  Базовый образ: официальная сборка PyTorch с CUDA 12.1
# ====================================================
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ffmpeg git wget libgl1 && \
    rm -rf /var/lib/apt/lists/*

# ===============================
# Рабочая директория
# ===============================
WORKDIR /app

# ===============================
# Установка Python-зависимостей
# ===============================
COPY requirements.txt .

# Обновляем pip и ставим зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir xformers==0.0.23.post1

# Проверка окружения
RUN python -c "import torch, xformers; print(f'✅ Torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, XFormers {xformers.__version__}')"

# ===============================
# Копируем приложение
# ===============================
COPY app.py .

# ===============================
# Порт и переменные окружения
# ===============================
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# ===============================
# Запуск FastAPI сервера
# ===============================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

