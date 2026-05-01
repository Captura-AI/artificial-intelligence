FROM python:3.11-slim

# System dependencies for OpenCV & EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download models during build (optional — remove to save image size and download at runtime)
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"
# RUN python -c "import clip; clip.load('ViT-B/32')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]