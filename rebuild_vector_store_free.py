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

def rebuild_vector_store_free_tier(data_path: str, output_dir: str = "data/vector_store"):
    """
    Rebuild FAISS vector store using Voyage AI Embeddings with FREE TIER rate limits
    Rate limits: 3 RPM (requests per minute), 10K TPM (tokens per minute)
    
    Strategy: Process 10 documents every 25 seconds = 2.4 RPM, ~200 tokens/doc = ~2400 TPM
    """
    start_time = time.time()
    print(f"🚀 Starting vector store rebuild with Voyage AI (FREE TIER MODE)...")
    print(f"📂 Data source: {data_path}")
    print(f"⚠️  FREE TIER LIMITS: 3 requests/min, 10K tokens/min")
    print(f"⚡ STRATEGY: 10 docs/batch, 25 sec delay = ~2.4 RPM, safe margin")
    
    # Check API Key
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("❌ VOYAGE_API_KEY not found in .env file")
    
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    print("\n📚 Loading processed documents...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_docs = len(data)
    print(f"✅ Loaded {total_docs:,} documents")
    
    # Prepare texts and metadatas
    texts = [doc['text'] for doc in data]
    metadatas = [doc['metadata'] for doc in data]
    
    model_name = "voyage-law-2"
    print(f"\n🧠 Using embedding model: {model_name}")
    print(f"⚡ Mode: Cloud API (Voyage AI) - FREE TIER")
    
    # Initialize Voyage Embeddings with VERY small batch size
    embeddings_model = VoyageAIEmbeddings(
        voyage_api_key=api_key,
        model=model_name,
        batch_size=10  # Very small to stay well under 10K TPM
    )
    
    # Create FAISS index with conservative rate limiting
    print("\n🔨 Building FAISS index with rate limiting...")
    
    vector_store = None
    chunk_size = 10  # Process only 10 docs at a time
    delay_between_requests = 25  # 25 seconds = 2.4 RPM (safe margin for 3 RPM limit)
    
    total_chunks = (len(texts) + chunk_size - 1) // chunk_size
    estimated_minutes = (total_chunks * delay_between_requests) / 60
    estimated_hours = estimated_minutes / 60
    
    print(f"\n📊 PROCESSING PLAN:")
    print(f"   • Total documents: {total_docs:,}")
    print(f"   • Batch size: {chunk_size} docs")
    print(f"   • Total batches: {total_chunks:,}")
    print(f"   • Delay per batch: {delay_between_requests} seconds")
    print(f"   • Estimated time: {estimated_minutes:.1f} minutes ({estimated_hours:.2f} hours)")
    print(f"\n⏰ Start time: {time.strftime('%H:%M:%S')}")
    print(f"📅 Expected completion: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_minutes * 60))}")
    
    input("\n⚠️  This will take ~{:.1f} hours. Press ENTER to start or Ctrl+C to cancel...".format(estimated_hours))
    
    print("\n")
    retry_count = 0
    max_retries = 3
    
    for i in tqdm(range(0, len(texts), chunk_size), total=total_chunks, desc="Indexing Batches", 
                  unit="batch", ncols=100):
        batch_texts = texts[i:i + chunk_size]
        batch_metadatas = metadatas[i:i + chunk_size]
        
        success = False
        for attempt in range(max_retries):
            try:
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
                
                success = True
                break  # Success, exit retry loop
                
            except Exception as e:
                if "RateLimitError" in str(type(e).__name__) or "rate limit" in str(e).lower():
                    retry_count += 1
                    wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                    print(f"\n⚠️  Rate limit hit at batch {i//chunk_size}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Non-rate-limit error
                    print(f"\n❌ Error at batch {i//chunk_size}: {e}")
                    print(f"💾 Saving progress...")
                    if vector_store:
                        output_path = Path(output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        vector_store.save_local(str(output_path / "faiss_index_partial"))
                        print(f"✅ Partial index saved ({i} docs): {output_path / 'faiss_index_partial'}")
                    raise
        
        if not success:
            raise Exception(f"Failed after {max_retries} retries at batch {i//chunk_size}")
        
        # Rate limiting: wait between requests (except after last batch)
        if i + chunk_size < len(texts):
            time.sleep(delay_between_requests)
    
    # Save vector store
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output_path / "faiss_index"))
    
    elapsed_time = time.time() - start_time
    elapsed_hours = elapsed_time / 3600
    
    print(f"\n✅ Vector store saved to {output_path / 'faiss_index'}")
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes ({elapsed_hours:.2f} hours)")
    print(f"📊 Documents indexed: {total_docs:,}")
    print(f"🔄 Rate limit retries: {retry_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild FAISS vector store with Voyage AI (Free Tier)")
    parser.add_argument("--data-path", type=str, default="data/processed/processed_cases.json", 
                       help="Path to processed JSON data")
    
    args = parser.parse_args()
    
    rebuild_vector_store_free_tier(args.data_path)
