import os

# Must be set before any C-extension imports to prevent OpenMP duplicate-lib segfault on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sqlite3
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys

# Add the project root to sys.path to resolve 'app' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# torch-based services must be imported before faiss — faiss-cpu loads its own BLAS/MKL,
# and if it runs before PyTorch the two runtimes conflict and cause a segfault on macOS.
from app.config import get_settings
from app.services.vehicle_detector import detect_vehicle, is_model_ready as yolo_ready
from app.services.plate_reader import read_license_plate, is_model_ready as ocr_ready
from app.services.clip_embedder import get_image_embedding, classify_scene_tags, is_model_ready as clip_ready

import faiss

def build_index(dataset_path: str, db_path: str, index_path: str, limit: int = 100):
    settings = get_settings()
    
    print("Warming up models...")
    yolo_ready(settings.yolo_model_path)
    ocr_ready(settings.ocr_languages)
    clip_ready(settings.clip_model_name)

    # Initialize SQLite DB
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            vehicle_type TEXT,
            license_plate TEXT,
            tags TEXT
        )
    """)
    conn.commit()

    # Find images
    image_paths = list(Path(dataset_path).rglob("*.jpg")) + list(Path(dataset_path).rglob("*.jpeg")) + list(Path(dataset_path).rglob("*.png"))
    image_paths = image_paths[:limit] # Limit for simulation

    if not image_paths:
        print(f"No images found in {dataset_path}")
        return

    # Initialize Faiss index (CLIP embedding size is 512)
    embedding_dim = 512
    index = faiss.IndexFlatIP(embedding_dim) # Inner product (Cosine Similarity since embeddings are usually normalized)

    print(f"Processing {len(image_paths)} images...")
    
    for i, path in enumerate(tqdm(image_paths)):
        try:
            image = Image.open(path).convert("RGB")
            
            # Detect vehicles
            detected_vehicles = detect_vehicle(
                image, 
                model_path=settings.yolo_model_path, 
                vehicle_class_ids=settings.vehicle_class_ids
            )
            
            # Read Plates
            vehicle_types = []
            plates = []
            for v_type, v_conf, bbox in detected_vehicles:
                vehicle_types.append(v_type)
                cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                plate, _ = read_license_plate(cropped, languages=settings.ocr_languages)
                if plate:
                    plates.append(plate)

            if not detected_vehicles:
                plate, _ = read_license_plate(image, languages=settings.ocr_languages)
                if plate:
                    plates.append(plate)

            vehicle_type_str = ",".join(vehicle_types) if vehicle_types else "OTHER"
            plate_str = ",".join(plates) if plates else None

            # Get Embedding
            embedding = get_image_embedding(image, model_name=settings.clip_model_name)
            
            # Classify Tags
            tags = classify_scene_tags(image, model_name=settings.clip_model_name)

            # Save to DB
            cursor.execute(
                "INSERT OR REPLACE INTO images (id, filename, vehicle_type, license_plate, tags) VALUES (?, ?, ?, ?, ?)",
                (i, str(path.absolute()), vehicle_type_str, plate_str, ",".join(tags))
            )
            
            # Add to Faiss
            emb_np = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(emb_np)
            index.add(emb_np)

        except Exception as e:
            print(f"Failed processing {path.name}: {e}")

    conn.commit()
    conn.close()
    faiss.write_index(index, index_path)
    print(f"Finished! Index saved to {index_path} and DB to {db_path}")

if __name__ == "__main__":
    DATASET_DIR = "app/datasets/images/test" # Using test split to simulate
    DB_PATH = "data/sim_metadata.db"
    INDEX_PATH = "data/sim_index.faiss"
    
    # Check if test dir exists, otherwise fallback
    if not os.path.exists(DATASET_DIR):
        DATASET_DIR = "app/datasets/images"
        
    build_index(DATASET_DIR, DB_PATH, INDEX_PATH, limit=100)
