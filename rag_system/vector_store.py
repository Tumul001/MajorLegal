# rag_system/vector_store.py
"""
Build and manage FAISS vector store for Indian legal documents
"""

import json
import numpy as np
import faiss
import os
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Import Voyage AI embeddings
try:
    from langchain_voyageai import VoyageAIEmbeddings
except ImportError:
    VoyageAIEmbeddings = None

# Import Google Generative AI embeddings
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    GoogleGenerativeAIEmbeddings = None

# Import Document from the correct module for newer LangChain versions
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain_community.docstore.document import Document
    except ImportError:
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class Document:
            page_content: str
            metadata: dict
            id: str = None

class IndianLegalVectorStore:
    """
    Vector store for Indian legal documents using FAISS
    """
    
    def __init__(self, embedding_model="text-embedding-3-small"):
        """
        Initialize vector store
        
        Args:
            embedding_model: Embedding model name (OpenAI or Google)
        """
        # Priority: Voyage AI > Google > OpenAI
        voyage_api_key = os.getenv("VOYAGE_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if voyage_api_key:
            print("ℹ️  Using Voyage AI embeddings (voyage-law-2)")
            if VoyageAIEmbeddings is None:
                raise ImportError(
                    "VoyageAIEmbeddings not available.\n"
                    "Install: pip install langchain-voyageai"
                )
            self.embeddings = VoyageAIEmbeddings(
                voyage_api_key=voyage_api_key,
                model="voyage-law-2"
            )
        elif google_api_key:
            print("ℹ️  Using Google Gemini embeddings")
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError(
                    "GoogleGenerativeAIEmbeddings not available.\n"
                    "Install: pip install langchain-google-genai"
                )
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_api_key
            )
        elif openai_api_key:
            print("ℹ️  Using OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=openai_api_key
            )
        else:
            raise ValueError(
                "No API key found in environment variables.\n"
                "Please create a .env file with:\n"
                "  VOYAGE_API_KEY=your_voyage_key (Recommended)\n"
                "  GOOGLE_API_KEY=your_google_api_key\n"
                "  OPENAI_API_KEY=your_openai_api_key"
            )
        
        self.vectorstore = None
        self.metadata_store = {}  # Store metadata separately
        
        # Paths
        self.index_dir = Path("data/vector_store")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.index_dir / "faiss_index"
        self.metadata_path = self.index_dir / "metadata.pkl"
    
    def build_from_processed_docs(self, processed_docs_path: str, batch_size: int = 32):
        """
        Build FAISS index from processed documents
        
        Args:
            processed_docs_path: Path to processed_cases.json
            batch_size: Number of documents to process at once
        """
        print("📚 Loading processed documents...")
        with open(processed_docs_path, 'r', encoding='utf-8') as f:
            processed_docs = json.load(f)
        
        print(f"✅ Loaded {len(processed_docs)} document chunks")
        
        # Convert to LangChain Document format
        print("🔄 Converting to LangChain documents...")
        documents = []
        for doc in processed_docs:
            metadata = doc.get('metadata', {})
            langchain_doc = Document(
                page_content=doc.get('text', ''),
                metadata={
                    'case_name': metadata.get('case_name', 'Unknown'),
                    'citation': metadata.get('citation', 'N/A'),
                    'court': metadata.get('court', 'Unknown Court'),
                    'date': metadata.get('date', 'Unknown'),
                    'judges': ', '.join(metadata.get('judges', [])) if isinstance(metadata.get('judges'), list) else str(metadata.get('judges', '')),
                    'acts_mentioned': ', '.join(metadata.get('acts_mentioned', [])) if isinstance(metadata.get('acts_mentioned'), list) else str(metadata.get('acts_mentioned', '')),
                    'sections': ', '.join(metadata.get('sections', [])) if isinstance(metadata.get('sections'), list) else str(metadata.get('sections', '')),
                    'url': metadata.get('url', ''),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'source': metadata.get('source', 'Indian Kanoon')
                }
            )
            documents.append(langchain_doc)
        
        print("🧠 Building FAISS index (this may take a while)...")
        print(f"📊 Total documents: {len(documents):,}")
        print(f"📦 Batch size: {batch_size}")
        
        # Build FAISS index in batches to avoid memory issues
        # Use larger initial batch
        initial_batch_size = min(batch_size * 2, 1000)
        print(f"🔨 Creating initial index with {initial_batch_size} documents...")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents[:initial_batch_size],
            embedding=self.embeddings
        )
        
        # Add remaining documents in batches with progress bar
        remaining_docs = documents[initial_batch_size:]
        total_batches = len(remaining_docs) // batch_size + (1 if len(remaining_docs) % batch_size else 0)
        print(f"➕ Adding remaining {len(remaining_docs):,} documents in {total_batches} batches...")
        
        for i in tqdm(range(0, len(remaining_docs), batch_size), desc="Building index"):
            batch = remaining_docs[i:i+batch_size]
            self.vectorstore.add_documents(batch)
            
            # NO checkpoints during build - save only at the end to avoid pickle bottleneck
        
        print("💾 Saving FAISS index...")
        self.save_index()
        
        print("✅ Vector store built successfully!")
        print(f"📊 Total documents indexed: {len(documents)}")
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        self.vectorstore.save_local(str(self.index_path))
        print(f"✅ Index saved to {self.index_path}")
    
    def load_index(self):
        """Load existing FAISS index from disk"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}")
        
        print("📂 Loading FAISS index...")
        self.vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True  # Required for loading
        )
        print("✅ Index loaded successfully!")
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Dict = None) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters (e.g., {'court': 'Supreme Court of India'})
        
        Returns:
            List of most similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Build or load an index first.")
        
        # Perform similarity search
        if filter_dict:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search with similarity scores
        
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        if self.vectorstore is None:
            return {"status": "Not initialized"}
        
        # Get total number of documents
        try:
            total_docs = self.vectorstore.index.ntotal
        except:
            total_docs = "Unknown"
        
        return {
            "total_documents": total_docs,
            "embedding_dimension": self.vectorstore.index.d if hasattr(self.vectorstore, 'index') else "Unknown",
            "index_path": str(self.index_path)
        }

# Usage
if __name__ == "__main__":
    # Build vector store from processed documents
    vector_store = IndianLegalVectorStore()
    
    # Option 1: Build new index
    vector_store.build_from_processed_docs(
        "data/processed/processed_cases.json",
        batch_size=32
    )
    
    # Option 2: Load existing index
    # vector_store.load_index()
    
    # Test search
    query = "What are the requirements for a valid arrest under CrPC?"
    results = vector_store.similarity_search(query, k=3)
    
    print("\\n🔍 Search Results:")
    for i, doc in enumerate(results, 1):
        print(f"\\n{i}. {doc.metadata['case_name']}")
        print(f"   Citation: {doc.metadata['citation']}")
        print(f"   Content: {doc.page_content[:200]}...")