import os

# Must be set before any C-extension imports to prevent OpenMP/BLAS conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sqlite3
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import sys

# Add the project root to sys.path to resolve 'app' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# torch-based services must be imported before faiss to establish PyTorch's BLAS first
from app.config import get_settings
from app.services.clip_embedder import get_text_embedding, is_model_ready as clip_ready

import faiss

# Configuration
DB_PATH = "data/sim_metadata.db"
INDEX_PATH = "data/sim_index.faiss"

st.set_page_config(page_title="Captura AI - Simulator", layout="wide")
st.title("📷 Captura AI - Search Simulator")

# Load Index & DB
@st.cache_resource
def load_data():
    if not os.path.exists(DB_PATH) or not os.path.exists(INDEX_PATH):
        return None, None
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    index = faiss.read_index(INDEX_PATH)
    return conn, index

conn, index = load_data()

if conn is None:
    st.error("Index not found! Please run `python -m app.scripts.build_index` first.")
    st.stop()

# Warming up CLIP for text search
settings = get_settings()
clip_ready(settings.clip_model_name)

# UI Modes
search_mode = st.radio("Search Mode", ["📝 License Plate (Exact/Partial)", "🧠 Semantic Visual (CLIP)"])
query = st.text_input("Enter your query...")

if st.button("Search") and query:
    if "License Plate" in search_mode:
        # DB Search
        df = pd.read_sql_query(f"SELECT * FROM images WHERE license_plate LIKE '%{query.upper()}%'", conn)
        
        if len(df) == 0:
            st.warning("No images found for this license plate.")
        else:
            st.success(f"Found {len(df)} images matching '{query.upper()}'")
            cols = st.columns(3)
            for i, row in df.iterrows():
                with cols[i % 3]:
                    img = Image.open(row['filename'])
                    st.image(img, caption=f"Plate: {row['license_plate']} | Vehicle: {row['vehicle_type']}")
                    
    else:
        # Semantic Search via CLIP
        text_emb = get_text_embedding(query, model_name=settings.clip_model_name)
        emb_np = np.array([text_emb], dtype=np.float32)
        faiss.normalize_L2(emb_np)
        
        # Search Faiss
        D, I = index.search(emb_np, k=6) # Top 6
        
        valid_indices = [idx for idx in I[0] if idx != -1]
        
        if not valid_indices:
            st.warning("No semantic matches found.")
        else:
            st.success(f"Top matches for '{query}'")
            
            # Fetch metadata
            placeholders = ','.join('?' * len(valid_indices))
            df = pd.read_sql_query(f"SELECT * FROM images WHERE id IN ({placeholders})", conn, params=valid_indices)
            
            # Sort dataframe by the order of indices returned by FAISS
            df['id_cat'] = pd.Categorical(df['id'], categories=valid_indices, ordered=True)
            df = df.sort_values('id_cat')
            
            cols = st.columns(3)
            for i, row in df.iterrows():
                with cols[i % 3]:
                    img = Image.open(row['filename'])
                    st.image(img, caption=f"Tags: {row['tags']} | Plate: {row['license_plate']}")
