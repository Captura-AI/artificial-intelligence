# 🌟 Captura AI Service

This project is built with FastAPI, Python, and integrates multiple AI models for image analysis.

## Starter Introduction

This application is an **AI microservice** for the Captura platform. It is designed to process street photography, extracting critical metadata such as vehicle types, license plates, GPS data, and CLIP embeddings. The architecture ensures fault tolerance across the AI pipeline stages, returning partial results even if one model fails.

---

## 🏆 Architecture Highlights

1. **Robust Pipeline Orchestration**: 
   - Downloads images and extracts EXIF metadata.
   - Detects the dominant vehicle type using YOLOv8.
   - Extracts Indonesian license plates via EasyOCR and regex fallbacks.
   - Generates 512-dim embeddings and zero-shot scene tags using CLIP ViT-B/32.

2. **Lazy Loading and Caching**:
   - Models are lazy-loaded upon the first request or warmed up during application startup.
   - Cached globally to prevent cold-start latency on subsequent requests.

3. **Flexible Configuration**:
   - Settings are centrally managed using `pydantic-settings`.
   - Confidence thresholds, OCR languages, and model paths can easily be overridden via environment variables.

## 📖 Notes

When models are first initialized, they will be downloaded automatically (YOLO ~6 MB, EasyOCR ~40 MB, CLIP ViT-B/32 ~350 MB). Ensure you have a stable internet connection on the first run.

## 🎖️ AI Technologies

| Technology | Description |
| ---------- | ----------- |
| FastAPI    | High-performance asynchronous web framework for Python. |
| YOLOv8     | State-of-the-art object detection model used to classify vehicle types (Car, Motorcycle, Bus, Truck, Bicycle). |
| EasyOCR    | Ready-to-use OCR with 80+ supported languages, tuned for Indonesian license plate recognition. |
| CLIP       | OpenAI's CLIP model (ViT-B/32) used for zero-shot scene tagging and generating semantic image embeddings. |

## 🏅 Dependencies & Libraries

| Library | Description | Version |
| ------- | ----------- | ------- |
| fastapi | Modern, fast web framework for building APIs. | 0.115.5 |
| pydantic | Data validation and settings management using Python type hints. | 2.10.3 |
| ultralytics | YOLOv8 object detection model implementation. | 8.3.54 |
| easyocr | End-to-End Multi-lingual Optical Character Recognition. | 1.7.2 |
| openai-clip | Official CLIP model by OpenAI for image-text embeddings. | 1.0.1 |
| streamlit | Python web app framework used for the local interactive simulator. | 1.31.0 |
| faiss-cpu | Library for efficient similarity search and clustering of dense vectors. | 1.8.0 |

## 🛠️ Setup Project

Follow these step-by-step instructions to get the AI service up and running in your development environment.

### 🍴 Prerequisites

Ensure these tools are pre-installed on your machine:
- [Python 3.10+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/) (Optional, for containerized deployment)
- [Git](https://git-scm.com/downloads)

## 🔍 Usage

### 🚀 Install Project

1. **Navigate to the AI Service Directory**
```bash
cd ai-service
```

2. **Create and Activate a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Variables**
Copy `.env.example` to `.env` before your first run.

5. **Run the Development Server**
```bash
uvicorn app.main:app --reload --port 8000
```

---

## 🎉 Build The App (Docker)

You can build and run the application via Docker for production or containerized environments.

```bash
docker build -t captura-ai .
docker run -p 8000:8000 --env-file .env captura-ai
```

## 🧪 Simulation & Scripts

The `app/scripts` folder contains utilities for building search indexes and running a local simulation UI to test the capabilities.

- **Build FAISS Index**:
```bash
python -m app.scripts.build_index
```

- **Run Simulator UI**:
```bash
streamlit run app/scripts/simulate_search.py
```

---

## 📂 Folder Structure

Project structure for the AI Service:

```text
ai-service
|   |_______app
|   |   |_______datasets         # Dataset loaders and processing
|   |   |_______models           # Pydantic schemas and types
|   |   |_______routers          # API endpoints and route definitions
|   |   |_______scripts          # Utility scripts (FAISS index builder, Streamlit simulator)
|   |   |_______services         # Core AI pipelines (YOLO, OCR, CLIP integration)
|   |   |_______config.py        # Centralized settings and configurations
|   |   |_______main.py          # FastAPI application entry point
|   |_______data                 # Local data directory for datasets and models
|   |_______requirements.txt     # Python dependencies
|   |_______Dockerfile           # Docker image definition
|   |_______CLAUDE.md            # AI Assistant instructions and context
```

### ⚒️ How to Contribute

Want to contribute? Great!
- Fork the repo
- Create a new branch (`git checkout -b improve-feature`)
- Make the appropriate changes in the files
- Commit your changes (`git commit -m 'feat: improve AI pipeline accuracy'`)
- Push to the branch (`git push origin improve-feature`)
- Create a Pull Request

## 📜 Credits

- Built with love for Captura AI platform by **@apiiyu**.
- Integrates YOLOv8 by Ultralytics, EasyOCR by Jaided AI, and CLIP by OpenAI.
