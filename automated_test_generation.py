"""
Automated Test Case Generation - No Human Annotation Required

Generates test cases with automatic ground truth extraction from existing legal cases
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import re


class AutomatedTestCaseGenerator:
    """Generate test cases automatically without human annotation"""
    
    def __init__(self, cases_file: str = "data/raw/merged_final_dataset.json"):
        self.cases_file = cases_file
        self.test_cases = []
    
    def load_cases(self) -> List[Dict]:
        """Load all cases from dataset"""
        with open(self.cases_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_citations_from_case(self, case_text: str) -> List[str]:
        """
        Automatically extract citations mentioned in case text
        
        Patterns matched:
        - 2020 SCC 123
        - AIR 1978 SC 597
        - [1978] 1 SCR 248
        """
        patterns = [
            r'\d{4}\s+SCC\s+\d+',  # 2020 SCC 123
            r'AIR\s+\d{4}\s+\w+\s+\d+',  # AIR 1978 SC 597
            r'\[\d{4}\]\s+\d+\s+\w+\s+\d+',  # [1978] 1 SCR 248
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, case_text, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    def generate_query_from_case(self, case: Dict) -> str:
        """
        Generate realistic legal query from case
        
        Extracts key legal issues mentioned in the case
        """
        text = case.get('data', {}).get('text', '') if 'data' in case else case.get('judgment_text', '')
        
        # Extract first paragraph or summary (usually contains main issue)
        paragraphs = text.split('\n\n')
        if paragraphs:
            # Take first substantial paragraph (>100 chars)
            for para in paragraphs[:5]:
                if len(para) > 100:
                    # Truncate to reasonable query length
                    query = para[:500].strip()
                    return query
        
        return text[:500].strip() if text else "Legal query about case"
    
    def create_ground_truth_relevance(self, query_case: Dict, 
                                     all_cases: List[Dict]) -> Dict:
        """
        Create ground truth relevance judgments automatically
        
        Method: Use case citations and text similarity
        - Highly relevant: Cases cited in the query case
        - Relevant: Cases with similar legal issues
        - Irrelevant: Random cases from different domains
        """
        query_citations = self.extract_citations_from_case(
            query_case.get('data', {}).get('text', '') if 'data' in query_case 
            else query_case.get('judgment_text', '')
        )
        
        # Create relevance levels
        ground_truth = {
            'highly_relevant': [],  # Cited cases
            'relevant': [],  # Same court/year
            'irrelevant': []  # Different domain
        }
        
        query_court = query_case.get('court', 'Unknown')
        query_year = query_case.get('date', '2020')[:4]
        
        for idx, case in enumerate(all_cases[:500]):  # Sample for efficiency
            case_id = f"doc_{idx}"
            case_text = case.get('data', {}).get('text', '') if 'data' in case else case.get('judgment_text', '')
            
            # Check if this case is cited in query case
            case_citations = self.extract_citations_from_case(case_text)
            if any(cit in query_citations for cit in case_citations):
                ground_truth['highly_relevant'].append(case_id)
            # Same court and similar time period
            elif (case.get('court') == query_court and 
                  abs(int(case.get('date', '2020')[:4]) - int(query_year)) <= 5):
                ground_truth['relevant'].append(case_id)
            # Different context
            else:
                ground_truth['irrelevant'].append(case_id)
        
        return ground_truth
    
    def generate_test_suite(self, num_cases: int = 50) -> List[Dict]:
        """
        Generate complete test suite with automatic ground truth
        
        Returns test cases ready for evaluation
        """
        print(f"Loading cases from {self.cases_file}...")
        all_cases = self.load_cases()
        print(f"Loaded {len(all_cases):,} cases")
        
        # Sample diverse cases
        sampled_cases = random.sample(all_cases, min(num_cases, len(all_cases)))
        
        test_cases = []
        for idx, case in enumerate(sampled_cases):
            print(f"Generating test case {idx+1}/{len(sampled_cases)}...", end='\r')
            
            # Generate query
            query = self.generate_query_from_case(case)
            
            # Create ground truth
            ground_truth = self.create_ground_truth_relevance(case, all_cases)
            
            test_case = {
                'id': f'test_case_{idx:03d}',
                'query': query,
                'ground_truth_relevant': (
                    ground_truth['highly_relevant'][:10] + 
                    ground_truth['relevant'][:10]
                ),
                'ground_truth_irrelevant': ground_truth['irrelevant'][:20],
                'source_case': {
                    'title': case.get('title', 'Unknown'),
                    'court': case.get('court', 'Unknown'),
                    'date': case.get('date', 'Unknown')
                },
                'metadata': {
                    'highly_relevant_count': len(ground_truth['highly_relevant']),
                    'relevant_count': len(ground_truth['relevant']),
                    'irrelevant_count': len(ground_truth['irrelevant'])
                }
            }
            
            test_cases.append(test_case)
        
        print(f"\n✅ Generated {len(test_cases)} test cases with automatic ground truth")
        return test_cases
    
    def save_test_suite(self, test_cases: List[Dict], 
                       output_file: str = "test_suite_automated.json"):
        """Save test suite to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        print(f"💾 Test suite saved to {output_file}")


class BaselineComparison:
    """Compare system against simple baselines (no human input needed)"""
    
    @staticmethod
    def random_baseline(query: str, all_docs: List[str], k: int = 5) -> List[str]:
        """Baseline 1: Random retrieval"""
        return random.sample(all_docs, min(k, len(all_docs)))
    
    @staticmethod
    def keyword_baseline(query: str, all_docs: List[Dict], k: int = 5) -> List[str]:
        """
        Baseline 2: Simple keyword matching (BM25-like)
        
        No external dependencies - just word overlap
        """
        query_words = set(query.lower().split())
        
        scores = []
        for idx, doc in enumerate(all_docs):
            doc_text = doc.get('data', {}).get('text', '') if 'data' in doc else doc.get('judgment_text', '')
            doc_words = set(doc_text.lower().split())
            
            # Simple overlap score
            overlap = len(query_words.intersection(doc_words))
            scores.append((f"doc_{idx}", overlap))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in scores[:k]]


# ============================================
# USAGE WITHOUT HUMAN ANNOTATION
# ============================================

def generate_evaluation_dataset():
    """
    Complete evaluation without human annotation
    
    Steps:
    1. Generate test cases from existing data
    2. Extract ground truth automatically
    3. Run system evaluation
    4. Compare against baselines
    5. Report metrics
    """
    generator = AutomatedTestCaseGenerator()
    
    # Generate 50 test cases
    test_cases = generator.generate_test_suite(num_cases=50)
    
    # Save for reuse
    generator.save_test_suite(test_cases)
    
    return test_cases


if __name__ == "__main__":
    print("\n" + "="*70)
    print("AUTOMATED TEST CASE GENERATION (NO HUMAN ANNOTATION)")
    print("="*70)
    print("\nThis approach:")
    print("✓ Uses existing case citations as ground truth")
    print("✓ No manual annotation required")
    print("✓ Generates 50+ test cases automatically")
    print("✓ Includes baseline comparisons")
    print("\n" + "="*70)
    
    test_cases = generate_evaluation_dataset()
    
    print("\n📊 Test Suite Statistics:")
    print(f"Total test cases: {len(test_cases)}")
    
    avg_relevant = sum(len(tc['ground_truth_relevant']) for tc in test_cases) / len(test_cases)
    avg_irrelevant = sum(len(tc['ground_truth_irrelevant']) for tc in test_cases) / len(test_cases)
    
    print(f"Avg relevant docs per case: {avg_relevant:.1f}")
    print(f"Avg irrelevant docs per case: {avg_irrelevant:.1f}")
    
    print("\n✅ Ready for evaluation without human annotation!")
    print("\nNext: Run evaluation_metrics.py with this test suite")
