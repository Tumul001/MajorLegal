"""
Simple Mock Embeddings for Testing
Uses hash-based deterministic random vectors - no sklearn, no complex libraries
"""
from typing import List
import hashlib
import numpy as np


class SimpleMockEmbeddings:
    """
    Ultra-simple mock embeddings using hash-based random vectors.
    No sklearn TF-IDF, no complex processing - just fast deterministic vectors.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize simple mock embeddings.
        
        Args:
            dimension: Embedding vector dimension (default 384 to match all-MiniLM-L6-v2)
        """
        self.dimension = dimension
    
    def _text_to_vector(self, text: str) -> List[float]:
        """
        Convert text to embedding vector using hash-based randomness.
        
        Args:
            text: Input text
            
        Returns:
            Normalized embedding vector
        """
        if not text or not isinstance(text, str):
            return [0.0] * self.dimension
        
        # Use MD5 hash to get deterministic seed from text
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)  # Use first 8 hex chars as seed
        
        # Generate deterministic random vector
        rng = np.random.RandomState(seed)
        vector = rng.randn(self.dimension)
        
        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query text.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        return self._text_to_vector(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embedding vectors
        """
        return [self._text_to_vector(text) for text in texts]
    
    def __call__(self, text: str) -> List[float]:
        """
        Make the object callable for compatibility with LangChain.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embed_query(text)
