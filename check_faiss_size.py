# check_faiss_size.py
"""Load the FAISS index and print the number of vectors (documents) stored."""
import os
import faiss
from pathlib import Path

INDEX_DIR = Path("data/vector_store/faiss_index")
INDEX_PATH = INDEX_DIR / "index.faiss"

if not INDEX_PATH.is_file():
    raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")

index = faiss.read_index(str(INDEX_PATH))
print(f"✅ FAISS index contains {index.ntotal} vectors (documents).")
