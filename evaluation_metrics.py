"""
Evaluation Metrics for Legal RAG System

Addresses:
1. RAG quality evaluation (Precision, Recall, F1, MRR, NDCG@k)
2. Citation verification and hallucination detection
3. Argument quality assessment
4. Human evaluation framework
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalRAGEvaluator:
    """Comprehensive evaluation suite for legal RAG system"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.evaluation_results = {
            'rag_metrics': {},
            'citation_verification': {},
            'hallucination_detection': {},
            'argument_quality': {}
        }
    
    # ============================================
    # 1. RAG RETRIEVAL METRICS
    # ============================================
    
    def calculate_precision_at_k(self, retrieved_docs: List[str], 
                                  relevant_docs: List[str], k: int = 5) -> float:
        """
        Precision@K: What fraction of top-k retrieved docs are relevant?
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of ground truth relevant document IDs
            k: Number of top documents to consider
        
        Returns:
            Precision score [0, 1]
        """
        top_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_set)
        return relevant_retrieved / k if k > 0 else 0.0
    
    def calculate_recall_at_k(self, retrieved_docs: List[str], 
                              relevant_docs: List[str], k: int = 5) -> float:
        """
        Recall@K: What fraction of relevant docs are in top-k?
        
        Returns:
            Recall score [0, 1]
        """
        top_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        relevant_retrieved = len(top_k.intersection(relevant_set))
        return relevant_retrieved / len(relevant_set) if relevant_set else 0.0
    
    def calculate_mrr(self, retrieved_docs: List[str], 
                     relevant_docs: List[str]) -> float:
        """
        Mean Reciprocal Rank: Position of first relevant document
        
        MRR = 1 / rank of first relevant doc
        Higher is better (1.0 = relevant doc at position 1)
        
        Returns:
            MRR score [0, 1]
        """
        relevant_set = set(relevant_docs)
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in relevant_set:
                return 1.0 / rank
        return 0.0
    
    def calculate_ndcg_at_k(self, retrieved_docs: List[str], 
                           relevance_scores: List[float], k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        
        Accounts for both relevance AND position
        NDCG = DCG / IDCG (normalized by ideal ranking)
        
        Args:
            retrieved_docs: Retrieved document IDs
            relevance_scores: Ground truth relevance scores (0-1 or 0-3)
            k: Number of documents to consider
        
        Returns:
            NDCG score [0, 1]
        """
        def dcg(scores, k):
            scores = scores[:k]
            return np.sum([
                (2**rel - 1) / np.log2(idx + 2)
                for idx, rel in enumerate(scores)
            ])
        
        # DCG of retrieved ranking
        dcg_score = dcg(relevance_scores[:k], k)
        
        # IDCG (ideal DCG with perfect ranking)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg_score = dcg(ideal_scores, k)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0
    
    # ============================================
    # 2. CITATION VERIFICATION
    # ============================================
    
    def verify_citation_relevance(self, argument_text: str, 
                                  case_excerpt: str, 
                                  threshold: float = 0.3) -> Dict:
        """
        Verify if citation actually supports the argument
        
        Uses semantic similarity between argument and case excerpt
        
        Args:
            argument_text: The legal argument text
            case_excerpt: Excerpt from cited case
            threshold: Minimum similarity score (0.3 = conservative)
        
        Returns:
            {
                'is_relevant': bool,
                'similarity_score': float,
                'confidence': str ('high', 'medium', 'low'),
                'warning': str or None
            }
        """
        # Encode texts
        arg_embedding = self.model.encode(argument_text, convert_to_tensor=True)
        case_embedding = self.model.encode(case_excerpt, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.cos_sim(arg_embedding, case_embedding).item()
        
        # Determine relevance and confidence
        is_relevant = similarity >= threshold
        
        if similarity >= 0.5:
            confidence = 'high'
            warning = None
        elif similarity >= 0.3:
            confidence = 'medium'
            warning = "Citation may be tangentially related - verify context"
        else:
            confidence = 'low'
            warning = "⚠️ CITATION MISMATCH: Case excerpt does not clearly support argument"
        
        return {
            'is_relevant': is_relevant,
            'similarity_score': float(similarity),
            'confidence': confidence,
            'warning': warning
        }
    
    def detect_hallucinated_citations(self, citations: List[Dict], 
                                     argument_text: str) -> List[Dict]:
        """
        Detect potentially hallucinated or misattributed citations
        
        Flags:
        - Citations with 'Unknown Case' or 'N/A'
        - Low semantic similarity to argument
        - Auto-generated citations (marked in metadata)
        
        Args:
            citations: List of citation dicts with case_name, excerpt, etc.
            argument_text: The argument text these citations support
        
        Returns:
            List of flagged citations with warnings
        """
        flagged_citations = []
        
        for idx, citation in enumerate(citations):
            flags = []
            
            # Check for placeholder values
            if citation.get('case_name') in ['Unknown Case', 'Unknown', None]:
                flags.append("PLACEHOLDER_NAME")
            
            if citation.get('citation') in ['N/A', None, '']:
                flags.append("MISSING_CITATION")
            
            if citation.get('auto_generated', False):
                flags.append("AUTO_GENERATED")
            
            # Check semantic relevance
            excerpt = citation.get('excerpt', '')
            if excerpt:
                verification = self.verify_citation_relevance(
                    argument_text, 
                    excerpt,
                    threshold=0.25  # Lower threshold for hallucination detection
                )
                
                if not verification['is_relevant']:
                    flags.append("LOW_RELEVANCE")
                    flags.append(f"similarity={verification['similarity_score']:.2f}")
            else:
                flags.append("MISSING_EXCERPT")
            
            if flags:
                flagged_citations.append({
                    'citation_index': idx,
                    'case_name': citation.get('case_name'),
                    'flags': flags,
                    'risk_level': 'HIGH' if len(flags) >= 3 else 'MEDIUM',
                    'recommendation': 'REMOVE' if 'PLACEHOLDER_NAME' in flags else 'VERIFY'
                })
        
        return flagged_citations
    
    # ============================================
    # 3. ARGUMENT QUALITY METRICS
    # ============================================
    
    def calculate_citation_density(self, argument: Dict) -> float:
        """
        Citations per 100 words of argument
        Healthy range: 2-5 citations per 100 words
        """
        text = argument.get('main_argument', '')
        word_count = len(text.split())
        citation_count = len(argument.get('case_citations', []))
        
        if word_count == 0:
            return 0.0
        
        return (citation_count / word_count) * 100
    
    def calculate_citation_quality_score(self, citations: List[Dict], 
                                        argument_text: str) -> Dict:
        """
        Comprehensive citation quality assessment
        
        Returns:
            {
                'overall_score': float [0, 1],
                'verified_citations': int,
                'hallucinated_citations': int,
                'avg_relevance': float,
                'quality_grade': str
            }
        """
        if not citations:
            return {
                'overall_score': 0.0,
                'verified_citations': 0,
                'hallucinated_citations': 0,
                'avg_relevance': 0.0,
                'quality_grade': 'F'
            }
        
        relevance_scores = []
        verified_count = 0
        hallucinated_count = 0
        
        for citation in citations:
            excerpt = citation.get('excerpt', '')
            if not excerpt:
                hallucinated_count += 1
                continue
            
            verification = self.verify_citation_relevance(argument_text, excerpt)
            relevance_scores.append(verification['similarity_score'])
            
            if verification['is_relevant'] and verification['confidence'] in ['high', 'medium']:
                verified_count += 1
            else:
                hallucinated_count += 1
        
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        overall_score = verified_count / len(citations)
        
        # Grade assignment
        if overall_score >= 0.9:
            grade = 'A'
        elif overall_score >= 0.75:
            grade = 'B'
        elif overall_score >= 0.5:
            grade = 'C'
        elif overall_score >= 0.25:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'overall_score': float(overall_score),
            'verified_citations': verified_count,
            'hallucinated_citations': hallucinated_count,
            'avg_relevance': float(avg_relevance),
            'quality_grade': grade
        }
    
    # ============================================
    # 4. HUMAN EVALUATION FRAMEWORK
    # ============================================
    
    def create_annotation_template(self, case_data: Dict) -> Dict:
        """
        Generate template for human annotators to evaluate arguments
        
        Returns JSON that annotators fill out
        """
        return {
            'case_id': case_data.get('id'),
            'scenario': case_data.get('scenario'),
            'annotations': {
                'prosecution_argument': {
                    'legal_soundness': None,  # 1-5 scale
                    'citation_accuracy': None,  # 1-5 scale
                    'persuasiveness': None,  # 1-5 scale
                    'notes': ""
                },
                'defense_argument': {
                    'legal_soundness': None,
                    'citation_accuracy': None,
                    'persuasiveness': None,
                    'notes': ""
                },
                'moderator_verdict': {
                    'agrees_with_verdict': None,  # True/False
                    'reasoning_quality': None,  # 1-5 scale
                    'expected_verdict': None,  # 'prosecution' or 'defense'
                    'notes': ""
                },
                'overall_quality': None,  # 1-5 scale
                'annotator_id': None,
                'timestamp': None
            }
        }
    
    def calculate_inter_annotator_agreement(self, annotations: List[Dict]) -> Dict:
        """
        Calculate Cohen's Kappa for inter-annotator agreement
        
        Measures reliability of human evaluations
        Kappa > 0.8 = strong agreement
        """
        # This is a simplified version - full implementation would use sklearn
        # For production, use: from sklearn.metrics import cohen_kappa_score
        
        logger.info("Inter-annotator agreement calculation requires multiple annotators")
        return {
            'cohen_kappa': None,  # Implement with sklearn
            'status': 'Requires multiple human annotations',
            'recommendation': 'Collect at least 2-3 annotations per case'
        }
    
    # ============================================
    # 5. COMPREHENSIVE EVALUATION REPORT
    # ============================================
    
    def generate_evaluation_report(self, test_cases: List[Dict]) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            test_cases: List of evaluated cases with ground truth
        
        Returns:
            Complete metrics report suitable for publication
        """
        report = {
            'metadata': {
                'total_cases': len(test_cases),
                'timestamp': None  # Add timestamp
            },
            'rag_metrics': {
                'avg_precision_at_5': [],
                'avg_recall_at_5': [],
                'avg_mrr': [],
                'avg_ndcg_at_5': []
            },
            'citation_quality': {
                'total_citations': 0,
                'verified_citations': 0,
                'hallucinated_citations': 0,
                'avg_citation_relevance': []
            },
            'hallucination_rate': 0.0,
            'argument_quality_distribution': {
                'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0
            }
        }
        
        logger.info(f"Generating evaluation report for {len(test_cases)} cases")
        
        for case in test_cases:
            # Calculate RAG metrics if ground truth available
            if 'ground_truth_docs' in case:
                report['rag_metrics']['avg_precision_at_5'].append(
                    self.calculate_precision_at_k(
                        case['retrieved_docs'], 
                        case['ground_truth_docs'], 
                        k=5
                    )
                )
            
            # Citation quality analysis
            for argument in [case.get('prosecution'), case.get('defense')]:
                if argument and 'case_citations' in argument:
                    quality = self.calculate_citation_quality_score(
                        argument['case_citations'],
                        argument.get('main_argument', '')
                    )
                    
                    report['citation_quality']['total_citations'] += len(argument['case_citations'])
                    report['citation_quality']['verified_citations'] += quality['verified_citations']
                    report['citation_quality']['hallucinated_citations'] += quality['hallucinated_citations']
                    report['citation_quality']['avg_citation_relevance'].append(quality['avg_relevance'])
                    report['argument_quality_distribution'][quality['quality_grade']] += 1
        
        # Calculate averages
        for metric in report['rag_metrics']:
            values = report['rag_metrics'][metric]
            report['rag_metrics'][metric] = {
                'mean': float(np.mean(values)) if values else 0.0,
                'std': float(np.std(values)) if values else 0.0
            }
        
        if report['citation_quality']['avg_citation_relevance']:
            report['citation_quality']['mean_relevance'] = float(
                np.mean(report['citation_quality']['avg_citation_relevance'])
            )
        
        # Hallucination rate
        total_cites = report['citation_quality']['total_citations']
        halluc_cites = report['citation_quality']['hallucinated_citations']
        report['hallucination_rate'] = (halluc_cites / total_cites) if total_cites > 0 else 0.0
        
        return report
    
    def save_report(self, report: Dict, output_path: str = "evaluation_results.json"):
        """Save evaluation report to file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Evaluation report saved to {output_path}")


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    evaluator = LegalRAGEvaluator()
    
    # Example: Verify a citation
    argument = "The accused has the right to remain silent under Article 20(3)"
    case_excerpt = "Article 20(3) states that no person accused of any offence shall be compelled to be a witness against himself"
    
    verification = evaluator.verify_citation_relevance(argument, case_excerpt)
    print("\n=== Citation Verification ===")
    print(f"Relevance: {verification['is_relevant']}")
    print(f"Similarity: {verification['similarity_score']:.3f}")
    print(f"Confidence: {verification['confidence']}")
    if verification['warning']:
        print(f"Warning: {verification['warning']}")
    
    # Example: Detect hallucinations
    citations = [
        {
            'case_name': 'Unknown Case',
            'citation': 'N/A',
            'excerpt': 'Some generic legal text here',
            'auto_generated': True
        },
        {
            'case_name': 'State v. Kumar',
            'citation': '2020 SCC 123',
            'excerpt': 'The court held that bail is the rule and jail is the exception',
            'auto_generated': False
        }
    ]
    
    flagged = evaluator.detect_hallucinated_citations(citations, argument)
    print("\n=== Hallucination Detection ===")
    for flag in flagged:
        print(f"Citation {flag['citation_index']}: {flag['flags']}")
        print(f"  Risk: {flag['risk_level']}, Action: {flag['recommendation']}")
