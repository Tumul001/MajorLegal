"""
Citation Verification System - Retrieval-Augmented Verification (RAV)
Prevents citation hallucination by verifying every citation against vector store
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from sentence_transformers import SentenceTransformer, util
import torch


@dataclass
class CitationVerification:
    """Result of citation verification"""
    is_verified: bool
    confidence: float  # 0.0 to 1.0
    verification_method: str
    matched_document: Optional[Dict] = None
    flag: str = "VERIFIED"  # VERIFIED, UNVERIFIED, UNCERTAIN, HALLUCINATED
    error_message: Optional[str] = None


class CitationVerifier:
    """
    Retrieval-Augmented Verification (RAV) System
    
    Verifies citations against vector store to prevent hallucination
    """
    
    def __init__(self, vector_store):
        """
        Initialize verifier with vector store
        
        Args:
            vector_store: IndianLegalVectorStore instance
        """
        self.vector_store = vector_store
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Verification thresholds
        self.HIGH_CONFIDENCE_THRESHOLD = 0.85  # Strong match
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.70  # Probable match
        self.LOW_CONFIDENCE_THRESHOLD = 0.50  # Weak match
    
    def verify_case_exists(self, case_name: str, citation: str = None, 
                          year: str = None) -> CitationVerification:
        """
        Verify if a case exists in the database
        
        Uses multiple verification strategies:
        1. Exact case name match in metadata
        2. Citation pattern match
        3. Semantic similarity search
        4. Year + partial name match
        
        Args:
            case_name: Name of the case (e.g., "Maneka Gandhi v. Union of India")
            citation: Legal citation (e.g., "AIR 1978 SC 597")
            year: Year of judgment
        
        Returns:
            CitationVerification with verification result
        """
        
        # Strategy 1: Direct metadata search via vector store
        if citation:
            result = self._verify_by_citation(case_name, citation)
            if result.is_verified:
                return result
        
        # Strategy 2: Semantic similarity search
        result = self._verify_by_semantic_search(case_name, year)
        if result.confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return result
        
        # Strategy 3: Partial name matching
        if year:
            result = self._verify_by_partial_match(case_name, year)
            if result.is_verified:
                return result
        
        # No verification successful
        return CitationVerification(
            is_verified=False,
            confidence=0.0,
            verification_method="all_methods_failed",
            flag="UNVERIFIED",
            error_message=f"Could not verify '{case_name}' in database"
        )
    
    def _verify_by_citation(self, case_name: str, citation: str) -> CitationVerification:
        """Verify by searching for citation pattern"""
        try:
            # Search for documents containing this citation
            query = f"citation {citation} {case_name}"
            results = self.vector_store.similarity_search(query, k=5)
            
            for doc in results:
                metadata = doc.metadata
                
                # Check if citation matches
                if citation.lower() in str(metadata.get('citation', '')).lower():
                    return CitationVerification(
                        is_verified=True,
                        confidence=1.0,
                        verification_method="exact_citation_match",
                        matched_document=metadata,
                        flag="VERIFIED"
                    )
            
            return CitationVerification(
                is_verified=False,
                confidence=0.0,
                verification_method="citation_search_failed",
                flag="UNVERIFIED"
            )
        
        except Exception as e:
            return CitationVerification(
                is_verified=False,
                confidence=0.0,
                verification_method="citation_search_error",
                flag="UNVERIFIED",
                error_message=str(e)
            )
    
    def _verify_by_semantic_search(self, case_name: str, year: str = None) -> CitationVerification:
        """Verify by semantic similarity to database cases"""
        try:
            # Build search query
            query = f"{case_name}"
            if year:
                query += f" {year}"
            
            # Retrieve similar cases
            results = self.vector_store.similarity_search_with_score(query, k=3)
            
            if not results:
                return CitationVerification(
                    is_verified=False,
                    confidence=0.0,
                    verification_method="no_semantic_matches",
                    flag="UNVERIFIED"
                )
            
            # Get best match
            best_doc, best_score = results[0]
            metadata = best_doc.metadata
            
            # Normalize score (FAISS uses distance, lower is better)
            # Convert to similarity (0-1 scale)
            confidence = max(0.0, min(1.0, 1.0 - best_score))
            
            # Verify case name similarity
            db_case_name = metadata.get('case_name', metadata.get('title', ''))
            name_similarity = self._calculate_name_similarity(case_name, db_case_name)
            
            # Combine scores
            final_confidence = (confidence + name_similarity) / 2
            
            # Determine verification status
            if final_confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
                flag = "VERIFIED"
                is_verified = True
            elif final_confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                flag = "VERIFIED"
                is_verified = True
            elif final_confidence >= self.LOW_CONFIDENCE_THRESHOLD:
                flag = "UNCERTAIN"
                is_verified = False
            else:
                flag = "UNVERIFIED"
                is_verified = False
            
            return CitationVerification(
                is_verified=is_verified,
                confidence=final_confidence,
                verification_method="semantic_similarity",
                matched_document=metadata if is_verified else None,
                flag=flag
            )
        
        except Exception as e:
            return CitationVerification(
                is_verified=False,
                confidence=0.0,
                verification_method="semantic_search_error",
                flag="UNVERIFIED",
                error_message=str(e)
            )
    
    def _verify_by_partial_match(self, case_name: str, year: str) -> CitationVerification:
        """Verify by partial name and year matching"""
        try:
            # Extract key terms from case name
            key_terms = self._extract_key_terms(case_name)
            
            # Search with key terms + year
            query = f"{' '.join(key_terms)} {year}"
            results = self.vector_store.similarity_search(query, k=5)
            
            for doc in results:
                metadata = doc.metadata
                db_case_name = metadata.get('case_name', metadata.get('title', ''))
                db_year = metadata.get('date', '')[:4] if metadata.get('date') else ''
                
                # Check if year matches and name has significant overlap
                if year == db_year:
                    name_sim = self._calculate_name_similarity(case_name, db_case_name)
                    if name_sim >= 0.6:  # 60% similarity threshold
                        return CitationVerification(
                            is_verified=True,
                            confidence=name_sim,
                            verification_method="partial_match_with_year",
                            matched_document=metadata,
                            flag="VERIFIED"
                        )
            
            return CitationVerification(
                is_verified=False,
                confidence=0.0,
                verification_method="no_partial_matches",
                flag="UNVERIFIED"
            )
        
        except Exception as e:
            return CitationVerification(
                is_verified=False,
                confidence=0.0,
                verification_method="partial_match_error",
                flag="UNVERIFIED",
                error_message=str(e)
            )
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two case names using embeddings"""
        try:
            # Encode both names
            embedding1 = self.model.encode(name1.lower(), convert_to_tensor=True)
            embedding2 = self.model.encode(name2.lower(), convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.cos_sim(embedding1, embedding2).item()
            
            return max(0.0, min(1.0, similarity))
        
        except Exception:
            # Fallback to simple string matching
            name1_clean = name1.lower().strip()
            name2_clean = name2.lower().strip()
            
            if name1_clean == name2_clean:
                return 1.0
            elif name1_clean in name2_clean or name2_clean in name1_clean:
                return 0.7
            else:
                return 0.0
    
    def _extract_key_terms(self, case_name: str) -> List[str]:
        """Extract key terms from case name"""
        # Remove common legal terms
        stop_words = {'v', 'vs', 'versus', 'state', 'union', 'government', 'of', 'india', 'the'}
        
        # Split and clean
        terms = re.findall(r'\w+', case_name.lower())
        key_terms = [t for t in terms if t not in stop_words and len(t) > 2]
        
        return key_terms[:5]  # Return top 5 key terms
    
    def verify_all_citations(self, citations: List[Dict]) -> List[Tuple[Dict, CitationVerification]]:
        """
        Verify all citations in a list
        
        Args:
            citations: List of citation dicts with 'case_name', 'citation', 'year'
        
        Returns:
            List of (citation, verification_result) tuples
        """
        results = []
        
        for citation in citations:
            case_name = citation.get('case_name', '')
            citation_ref = citation.get('citation', '')
            year = citation.get('year', '')
            
            # Skip empty citations
            if not case_name or case_name == 'Unknown Case':
                verification = CitationVerification(
                    is_verified=False,
                    confidence=0.0,
                    verification_method="empty_citation",
                    flag="HALLUCINATED",
                    error_message="Empty or placeholder case name"
                )
            else:
                verification = self.verify_case_exists(case_name, citation_ref, year)
            
            results.append((citation, verification))
        
        return results
    
    def get_verification_summary(self, verifications: List[Tuple[Dict, CitationVerification]]) -> Dict:
        """
        Generate summary statistics for verification results
        
        Returns:
            Dict with counts and percentages
        """
        total = len(verifications)
        if total == 0:
            return {
                'total': 0,
                'verified': 0,
                'unverified': 0,
                'uncertain': 0,
                'hallucinated': 0,
                'verification_rate': 0.0,
                'avg_confidence': 0.0
            }
        
        verified = sum(1 for _, v in verifications if v.flag == "VERIFIED")
        unverified = sum(1 for _, v in verifications if v.flag == "UNVERIFIED")
        uncertain = sum(1 for _, v in verifications if v.flag == "UNCERTAIN")
        hallucinated = sum(1 for _, v in verifications if v.flag == "HALLUCINATED")
        
        avg_confidence = sum(v.confidence for _, v in verifications) / total
        
        return {
            'total': total,
            'verified': verified,
            'unverified': unverified,
            'uncertain': uncertain,
            'hallucinated': hallucinated,
            'verification_rate': verified / total * 100,
            'avg_confidence': avg_confidence,
            'status': 'SAFE' if verified >= total * 0.8 else 'RISKY'
        }
