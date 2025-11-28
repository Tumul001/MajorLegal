import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def rebuild_vector_store(data_path: str, output_dir: str = "data/vector_store", batch_size: int = 100):
    """
    Rebuild FAISS vector store using Voyage AI Embeddings (API-based)
    """
    start_time = time.time()
    print(f"🚀 Starting vector store rebuild with Voyage AI...")
    print(f"📂 Data source: {data_path}")
    
    # Check API Key
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("❌ VOYAGE_API_KEY not found in .env file")
    
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    print("📚 Loading processed documents...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} documents")
    
    # Prepare texts and metadatas
    texts = [doc['text'] for doc in data]
    metadatas = [doc['metadata'] for doc in data]
    
    model_name = "voyage-law-2"
    print(f"🧠 Using embedding model: {model_name}")
    print(f"⚡ Mode: Cloud API (Voyage AI)")
    
    # Initialize Voyage Embeddings
    embeddings_model = VoyageAIEmbeddings(
        voyage_api_key=api_key,
        model=model_name,
        batch_size=batch_size
    )
    
    # Create FAISS index
    # Note: FAISS.from_texts automatically handles batching for embeddings
    print("🔨 Building FAISS index (this sends data to Voyage AI)...")
    
    # We'll do it in chunks to show progress and handle potential network issues gracefully
    vector_store = None
    chunk_size = 1000  # Process 1000 docs at a time
    
    total_chunks = (len(texts) + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(0, len(texts), chunk_size), total=total_chunks, desc="Indexing Batches"):
        batch_texts = texts[i:i + chunk_size]
        batch_metadatas = metadatas[i:i + chunk_size]
        
        if vector_store is None:
            vector_store = FAISS.from_texts(
                texts=batch_texts,
                embedding=embeddings_model,
                metadatas=batch_metadatas
            )
        else:
            vector_store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas
            )
            
        # Optional: Sleep briefly to be nice to the API rate limit if needed
        # time.sleep(0.1)
    
    # Save vector store
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output_path / "faiss_index"))
    
    elapsed_time = time.time() - start_time
    print(f"✅ Vector store saved to {output_path / 'faiss_index'}")
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild FAISS vector store with Voyage AI")
    parser.add_argument("--data-path", type=str, default="data/processed/processed_cases.json", help="Path to processed JSON data")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for API calls")
    
    args = parser.parse_args()
    
    rebuild_vector_store(args.data_path, batch_size=args.batch_size)
