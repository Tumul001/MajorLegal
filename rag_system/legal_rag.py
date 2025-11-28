# rag_system/legal_rag.py
"""
Production RAG system for legal debates using FAISS with Graph-RAG and Validation
"""

from typing import List, Dict, Tuple
import os
from pathlib import Path
import re

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

# Import Graph-RAG and Validation components
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from graph_manager import CitationGraph
    from legal_validator import LegalValidator
    GRAPH_RAG_AVAILABLE = True
except ImportError:
    GRAPH_RAG_AVAILABLE = False
    print("⚠️ Graph-RAG components not available - falling back to vector-only retrieval")


class ProductionLegalRAGSystem:
    """
    Production-ready RAG system for Indian legal documents with:
    - Vector similarity search (Voyage AI Law embeddings)
    - Citation graph analysis (PageRank)
    - Legal precedent validation (Shepardizing)
    """
    
    def __init__(self, index_path: str = None, use_graph_rag: bool = True):
        """
        Initialize RAG system
        
        Args:
            index_path: Path to FAISS index directory
            use_graph_rag: Whether to use Graph-RAG hybrid retrieval
        """
        self.vector_store = IndianLegalVectorStore()
        self.use_graph_rag = use_graph_rag and GRAPH_RAG_AVAILABLE
        
        # Load existing index
        try:
            self.vector_store.load_index()
            print("✅ RAG system initialized with existing index")
        except FileNotFoundError:
            print("⚠️ No existing index found. Please build one first.")
            raise
        
        # Initialize Graph-RAG components if available
        if self.use_graph_rag:
            try:
                self.citation_graph = CitationGraph()
                self.citation_graph.load_graph()
                print("✅ Citation graph loaded - Graph-RAG enabled")
            except FileNotFoundError:
                print("⚠️ Citation graph not found - using vector-only retrieval")
                self.use_graph_rag = False
                self.citation_graph = None
        else:
            self.citation_graph = None
        
        # Initialize legal validator
        if GRAPH_RAG_AVAILABLE:
            self.validator = LegalValidator()
            print("✅ Legal validator initialized")
        else:
            self.validator = None
    
    def retrieve_documents(self, query: str, k: int = 5, validate: bool = True) -> List[Document]:
        """
        Retrieve relevant documents using hybrid Graph-RAG or vector-only
        
        Args:
            query: Legal question or case description
            k: Number of documents to retrieve
            validate: Whether to validate precedents
        
        Returns:
            List of Document objects with metadata (including risk_level if validated)
        """
        if self.use_graph_rag and self.citation_graph:
            return self._retrieve_hybrid(query, k, validate)
        else:
            return self._retrieve_vector_only(query, k, validate)
    
    def _retrieve_vector_only(self, query: str, k: int, validate: bool) -> List[Document]:
        """Standard vector similarity search with smart ranking"""
        # Get candidates with scores
        doc_score_pairs = self.vector_store.similarity_search_with_score(query, k=k*2)
        
        # Apply smart ranking
        reranked_docs = self._apply_smart_ranking(doc_score_pairs)
        
        # Take top k
        top_docs = reranked_docs[:k]
        
        # Add validation if enabled
        if validate and self.validator:
            top_docs = self._validate_documents(top_docs)
        
        return top_docs
    
    def _retrieve_hybrid(self, query: str, k: int, validate: bool) -> List[Document]:
        """
        Hybrid retrieval combining vector similarity, PageRank, and Smart Ranking
        """
        # Get more candidates than needed (for re-ranking)
        candidate_docs = self.vector_store.similarity_search_with_score(query, k=k*2)
        
        # Compute hybrid scores first
        hybrid_candidates = []
        for doc, vector_score in candidate_docs:
            case_name = doc.metadata.get('case_name', '')
            
            # Get PageRank score (default 0.0 if case not in graph)
            pagerank_score = self.citation_graph.get_pagerank_score(case_name)
            
            # Hybrid score: 70% vector, 30% PageRank
            # Note: vector_score from Voyage/FAISS is typically cosine similarity (0-1)
            hybrid_score = (vector_score * 0.7) + (pagerank_score * 0.3)
            
            doc.metadata['vector_score'] = vector_score
            doc.metadata['pagerank_score'] = pagerank_score
            
            hybrid_candidates.append((doc, hybrid_score))
            
        # Apply Smart Ranking (Court Hierarchy + Recency) on top of Hybrid Score
        reranked_docs = self._apply_smart_ranking(hybrid_candidates)
        
        # Take top k
        top_docs = reranked_docs[:k]
        
        # Add validation if enabled
        if validate and self.validator:
            top_docs = self._validate_documents(top_docs)
        
        return top_docs

    def _apply_smart_ranking(self, doc_score_pairs: List[Tuple[Document, float]]) -> List[Document]:
        """
        Apply smart ranking based on court hierarchy and recency.
        
        Boosts scores for:
        1. Supreme Court cases (+20%)
        2. High Court cases (+10%)
        3. Recent cases (>2020: +15%, >2010: +10%, >2000: +5%)
        4. Statutes & Constitution (+25%)
        """
        reranked = []
        
        for doc, score in doc_score_pairs:
            metadata = doc.metadata
            boost_multiplier = 1.0
            
            # 1. Court Hierarchy
            court = str(metadata.get('court', '')).lower()
            if 'supreme court' in court:
                boost_multiplier += 0.20
            elif 'high court' in court:
                boost_multiplier += 0.10
                
            # 2. Recency
            date_str = str(metadata.get('date', ''))
            year = 0
            try:
                # Try to extract year (simple heuristic)
                match = re.search(r'\b(19|20)\d{2}\b', date_str)
                if match:
                    year = int(match.group(0))
            except:
                pass
                
            if year >= 2020:
                boost_multiplier += 0.15
            elif year >= 2010:
                boost_multiplier += 0.10
            elif year >= 2000:
                boost_multiplier += 0.05
            
            # 3. Source Boost (Statutes & Constitution)
            source = str(metadata.get('source', '')).lower()
            if 'statute' in source or 'constitution' in source:
                boost_multiplier += 0.25
                
                # Fix title for statutes if missing (IN-PLACE UPDATE)
                title = metadata.get('case_name', 'Unknown')
                if not title or title == 'Unknown':
                    act = metadata.get('act', 'Statute')
                    section = metadata.get('section', '?')
                    doc.metadata['case_name'] = f"{act} - Section {section}"
            
            # Apply boost
            final_score = score * boost_multiplier
            
            # Store scores for debugging/UI
            doc.metadata['original_score'] = score
            doc.metadata['smart_score'] = final_score
            doc.metadata['ranking_boost'] = boost_multiplier
            doc.metadata['year_extracted'] = year if year > 0 else "Unknown"
            
            reranked.append((doc, final_score))
            
        # Sort by new score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the documents
        return [doc for doc, score in reranked]
    
    def _validate_documents(self, docs: List[Document]) -> List[Document]:
        """
        Validate documents for legal precedent issues
        
        Adds risk_level, validation_flags, and validation_warnings to metadata
        """
        for doc in docs:
            case_text = doc.page_content
            validation = self.validator.validate_precedent(case_text)
            
            # Add validation metadata
            doc.metadata['risk_level'] = validation.risk_level
            doc.metadata['validation_flags'] = validation.flags
            doc.metadata['validation_warnings'] = validation.warnings
            doc.metadata['validation_confidence'] = validation.confidence
        
        return docs
    
    def retrieve_relevant_cases(
        self, 
        query: str, 
        k: int = 5,
        court_filter: str = None,
        min_relevance_score: float = 0.5,
        use_graph: bool = None
    ) -> List[str]:
        """
        Retrieve relevant case law for a query (formatted strings)
        
        Args:
            query: Legal question or case description
            k: Number of cases to retrieve
            court_filter: Optional court filter (e.g., "Supreme Court of India")
            min_relevance_score: Minimum similarity score threshold
            use_graph: Override Graph-RAG setting for this query
        
        Returns:
            List of formatted case excerpts
        """
        # Override Graph-RAG setting if specified
        if use_graph is not None:
            original_setting = self.use_graph_rag
            self.use_graph_rag = use_graph and self.citation_graph is not None
        
        # Retrieve documents
        docs = self.retrieve_documents(query, k=k)
        
        # Restore original setting
        if use_graph is not None:
            self.use_graph_rag = original_setting
        
        # Format results
        formatted_cases = []
        for doc in docs:
            formatted_case = self._format_case_with_validation(doc)
            formatted_cases.append(formatted_case)
        
        return formatted_cases
    
    def _format_case(self, doc: Document, score: float) -> str:
        """
        Format a document into a readable case summary (legacy method)
        
        Args:
            doc: LangChain document
            score: Relevance score
        
        Returns:
            Formatted case string
        """
        metadata = doc.metadata
        
        formatted = f"""
**{metadata.get('case_name', 'Unknown')}** ({metadata.get('citation', 'N/A')})
Court: {metadata.get('court', 'Unknown')} | Date: {metadata.get('date', 'Unknown')}
Relevance Score: {score:.2f}

Judges: {metadata.get('judges', 'N/A')}
Acts Mentioned: {metadata.get('acts_mentioned', 'N/A')}
Sections: {metadata.get('sections', 'N/A')}

Excerpt:
{doc.page_content}

---
"""
        return formatted
    
    def _format_case_with_validation(self, doc: Document) -> str:
        """
        Format a document with validation warnings
        
        Args:
            doc: LangChain document (with validation metadata)
        
        Returns:
            Formatted case string with risk warnings
        """
        metadata = doc.metadata
        
        # Build risk warning
        risk_warning = ""
        if metadata.get('risk_level') == 'HIGH':
            risk_warning = "\n🚨  **HIGH RISK**: This case may have been overruled or reversed.\n"
            risk_warning += "   " + "\n   ".join(metadata.get('validation_warnings', [])) + "\n"
        elif metadata.get('risk_level') == 'MEDIUM':
            risk_warning = "\n⚠️  **CAUTION**: This case may have been distinguished.\n"
            risk_warning += "   " + "\n   ".join(metadata.get('validation_warnings', [])) + "\n"
        
        # Build score info
        score_info = ""
        if 'smart_score' in metadata:
            score_info = f"Smart Score: {metadata['smart_score']:.3f} (Base: {metadata.get('original_score', 0):.3f} | Boost: {metadata.get('ranking_boost', 1.0):.2f}x)\n"
        elif 'hybrid_score' in metadata:
            score_info = f"Hybrid Score: {metadata['hybrid_score']:.3f} (Vector: {metadata['vector_score']:.3f} | PageRank: {metadata['pagerank_score']:.3f})\n"
        
        # Build title (Improved for Statutes)
        title = metadata.get('case_name', 'Unknown')
        if (not title or title == 'Unknown') and str(metadata.get('source', '')).lower() == 'statute':
            title = f"{metadata.get('act', 'Statute')} - Section {metadata.get('section', '')}"

        formatted = f"""
**{title}** ({metadata.get('citation', 'N/A')})
Court: {metadata.get('court', 'Unknown')} | Date: {metadata.get('date', 'Unknown')}
{score_info}{risk_warning}
Judges: {metadata.get('judges', 'N/A')}
Acts Mentioned: {metadata.get('acts_mentioned', 'N/A')}
Sections: {metadata.get('sections', 'N/A')}

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
        return [self._format_case_with_validation(doc) for doc in results]
    
    def search_by_section(self, section: str, act: str, k: int = 5) -> List[str]:
        """
        Search for cases related to a specific section
        
        Example: search_by_section("302", "IPC")
        """
        query = f"Section {section} of {act}"
        return self.retrieve_relevant_cases(query, k=k)
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics"""
        stats = self.vector_store.get_stats()
        
        if self.citation_graph:
            graph_stats = self.citation_graph.get_stats()
            stats.update({
                "graph_enabled": True,
                "citation_graph": graph_stats
            })
        else:
            stats["graph_enabled"] = False
        
        stats["validation_enabled"] = self.validator is not None
        
        return stats