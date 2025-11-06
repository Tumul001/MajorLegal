# rag_system/legal_rag.py
"""
Production RAG system for legal debates using FAISS
"""

from typing import List, Dict

# Import Document from the correct module for newer LangChain versions
try:
    from langchain_core.documents import Document
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict
        id: str = None

from .vector_store import IndianLegalVectorStore

class ProductionLegalRAGSystem:
    """
    Production-ready RAG system for Indian legal documents
    """
    
    def __init__(self, index_path: str = None):
        """
        Initialize RAG system
        
        Args:
            index_path: Path to FAISS index directory
        """
        self.vector_store = IndianLegalVectorStore()
        
        # Load existing index
        try:
            self.vector_store.load_index()
            print("✅ RAG system initialized with existing index")
        except FileNotFoundError:
            print("⚠️ No existing index found. Please build one first.")
            raise
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents (raw) for LangChain integration
        
        Args:
            query: Legal question or case description
            k: Number of documents to retrieve
        
        Returns:
            List of Document objects with metadata
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def retrieve_relevant_cases(
        self, 
        query: str, 
        k: int = 5,
        court_filter: str = None,
        min_relevance_score: float = 0.5
    ) -> List[str]:
        """
        Retrieve relevant case law for a query
        
        Args:
            query: Legal question or case description
            k: Number of cases to retrieve
            court_filter: Optional court filter (e.g., "Supreme Court of India")
            min_relevance_score: Minimum similarity score threshold
        
        Returns:
            List of formatted case excerpts
        """
        # Build filter dictionary
        filter_dict = {}
        if court_filter:
            filter_dict['court'] = court_filter
        
        # Perform search with scores
        results_with_scores = self.vector_store.similarity_search_with_score(
            query, 
            k=k * 2  # Get more results to filter by score
        )
        
        # Filter by relevance score
        filtered_results = [
            (doc, score) for doc, score in results_with_scores 
            if score >= min_relevance_score
        ][:k]
        
        # Format results
        formatted_cases = []
        for doc, score in filtered_results:
            formatted_case = self._format_case(doc, score)
            formatted_cases.append(formatted_case)
        
        return formatted_cases
    
    def _format_case(self, doc: Document, score: float) -> str:
        """
        Format a document into a readable case summary
        
        Args:
            doc: LangChain document
            score: Relevance score
        
        Returns:
            Formatted case string
        """
        metadata = doc.metadata
        
        formatted = f"""
**{metadata['case_name']}** ({metadata['citation']})
Court: {metadata['court']} | Date: {metadata['date']}
Relevance Score: {score:.2f}

Judges: {metadata['judges']}
Acts Mentioned: {metadata['acts_mentioned']}
Sections: {metadata['sections']}

Excerpt:
{doc.page_content}

---
"""
        return formatted
    
    def get_similar_cases_by_citation(self, case_citation: str, k: int = 3) -> List[str]:
        """Find cases similar to a given case"""
        results = self.vector_store.similarity_search(
            f"Cases similar to {case_citation}",
            k=k
        )
        return [self._format_case(doc, 1.0) for doc in results]
    
    def search_by_section(self, section: str, act: str, k: int = 5) -> List[str]:
        """
        Search for cases related to a specific section
        
        Example: search_by_section("302", "IPC")
        """
        query = f"Section {section} of {act}"
        return self.retrieve_relevant_cases(query, k=k)
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics"""
        return self.vector_store.get_stats()