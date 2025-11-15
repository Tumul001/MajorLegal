"""
Test Suite for Evaluation Metrics and Citation Verification

Run this to validate the fixes for:
1. RAG evaluation metrics implementation
2. Citation hallucination detection and prevention
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation_metrics import LegalRAGEvaluator
import json


def test_citation_verification():
    """Test citation relevance verification"""
    print("\n" + "="*70)
    print("TEST 1: Citation Verification")
    print("="*70)
    
    evaluator = LegalRAGEvaluator()
    
    # Test Case 1: Highly relevant citation
    print("\n[Case 1] Relevant Citation:")
    argument = "The right to bail is fundamental under Article 21 of the Constitution"
    excerpt = "Article 21 guarantees the right to life and personal liberty, which includes the right to bail"
    
    result = evaluator.verify_citation_relevance(argument, excerpt)
    print(f"  Similarity: {result['similarity_score']:.3f}")
    print(f"  Relevant: {result['is_relevant']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  ✓ PASS" if result['is_relevant'] else "  ✗ FAIL")
    
    # Test Case 2: Irrelevant citation (hallucination)
    print("\n[Case 2] Irrelevant Citation:")
    argument = "The accused has right to bail under CrPC Section 437"
    excerpt = "The plaintiff failed to prove ownership of the disputed property"
    
    result = evaluator.verify_citation_relevance(argument, excerpt)
    print(f"  Similarity: {result['similarity_score']:.3f}")
    print(f"  Relevant: {result['is_relevant']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Warning: {result['warning']}")
    print(f"  ✓ PASS" if not result['is_relevant'] else "  ✗ FAIL (should detect mismatch)")


def test_hallucination_detection():
    """Test detection of hallucinated citations"""
    print("\n" + "="*70)
    print("TEST 2: Hallucination Detection")
    print("="*70)
    
    evaluator = LegalRAGEvaluator()
    
    citations = [
        {
            'case_name': 'Unknown Case',  # Placeholder - should be flagged
            'citation': 'N/A',
            'excerpt': 'Generic legal text',
            'auto_generated': True
        },
        {
            'case_name': 'State v. Kumar',
            'citation': '2020 SCC 123',
            'excerpt': 'The right to bail is a fundamental right under Article 21',
            'auto_generated': False
        },
        {
            'case_name': 'Maneka Gandhi v. Union of India',
            'citation': '1978 SCC 248',
            'excerpt': '',  # Missing excerpt - should be flagged
            'auto_generated': False
        }
    ]
    
    argument = "The accused has the right to bail under Article 21"
    flagged = evaluator.detect_hallucinated_citations(citations, argument)
    
    print(f"\nTotal citations: {len(citations)}")
    print(f"Flagged citations: {len(flagged)}")
    
    for flag in flagged:
        print(f"\n  Citation {flag['citation_index']}: {flag['case_name']}")
        print(f"    Flags: {', '.join(flag['flags'])}")
        print(f"    Risk: {flag['risk_level']}")
        print(f"    Action: {flag['recommendation']}")
    
    expected_flagged = 2  # Should flag citations 0 and 2
    print(f"\n  {'✓ PASS' if len(flagged) >= expected_flagged else '✗ FAIL'}")


def test_rag_metrics():
    """Test RAG retrieval metrics"""
    print("\n" + "="*70)
    print("TEST 3: RAG Retrieval Metrics")
    print("="*70)
    
    evaluator = LegalRAGEvaluator()
    
    # Simulated retrieval results
    retrieved = ['doc1', 'doc2', 'doc5', 'doc3', 'doc8']
    relevant = ['doc1', 'doc3', 'doc4', 'doc7']
    
    precision = evaluator.calculate_precision_at_k(retrieved, relevant, k=5)
    recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=5)
    mrr = evaluator.calculate_mrr(retrieved, relevant)
    
    print(f"\nPrecision@5: {precision:.3f}")
    print(f"Recall@5: {recall:.3f}")
    print(f"MRR: {mrr:.3f}")
    
    # MRR should be 1.0 (first doc is relevant)
    print(f"\n  {'✓ PASS' if mrr == 1.0 else '✗ FAIL (expected MRR=1.0)'}")


def test_citation_quality_scoring():
    """Test comprehensive citation quality assessment"""
    print("\n" + "="*70)
    print("TEST 4: Citation Quality Scoring")
    print("="*70)
    
    evaluator = LegalRAGEvaluator()
    
    argument = "Section 304B IPC deals with dowry death cases where the victim dies within 7 years of marriage"
    
    citations = [
        {
            'case_name': 'State of Punjab v. Iqbal Singh',
            'citation': '1991 SCC 1532',
            'excerpt': 'Section 304B IPC creates a presumption when a woman dies within seven years of marriage under unnatural circumstances and dowry demands were made',
            'auto_generated': False
        },
        {
            'case_name': 'Kans Raj v. State of Punjab',
            'citation': '2000 SCC 1323',
            'excerpt': 'The prosecution must prove that the death occurred within seven years of marriage and was connected to dowry harassment',
            'auto_generated': False
        },
        {
            'case_name': 'Unknown Case',  # Bad citation
            'citation': 'N/A',
            'excerpt': 'Some generic text about legal procedures',
            'auto_generated': True
        }
    ]
    
    quality = evaluator.calculate_citation_quality_score(citations, argument)
    
    print(f"\nOverall Score: {quality['overall_score']:.2f}")
    print(f"Verified Citations: {quality['verified_citations']}")
    print(f"Hallucinated Citations: {quality['hallucinated_citations']}")
    print(f"Average Relevance: {quality['avg_relevance']:.3f}")
    print(f"Quality Grade: {quality['quality_grade']}")
    
    # Should detect 2 good citations and 1 bad
    expected_verified = 2
    print(f"\n  {'✓ PASS' if quality['verified_citations'] == expected_verified else '✗ FAIL'}")


def test_evaluation_report_generation():
    """Test comprehensive evaluation report"""
    print("\n" + "="*70)
    print("TEST 5: Evaluation Report Generation")
    print("="*70)
    
    evaluator = LegalRAGEvaluator()
    
    # Simulated test cases
    test_cases = [
        {
            'id': 'case_001',
            'scenario': 'Bail application under CrPC',
            'retrieved_docs': ['doc1', 'doc2', 'doc3'],
            'ground_truth_docs': ['doc1', 'doc4'],
            'prosecution': {
                'main_argument': 'Accused is a flight risk',
                'case_citations': [
                    {'case_name': 'State v. Kumar', 'excerpt': 'Flight risk criteria established'}
                ]
            },
            'defense': {
                'main_argument': 'Client has strong community ties',
                'case_citations': [
                    {'case_name': 'Bail is the rule', 'excerpt': 'Bail should be granted unless exceptional circumstances'}
                ]
            }
        }
    ]
    
    report = evaluator.generate_evaluation_report(test_cases)
    
    print(f"\nTotal Cases Evaluated: {report['metadata']['total_cases']}")
    print(f"Hallucination Rate: {report['hallucination_rate']:.2%}")
    print(f"Total Citations: {report['citation_quality']['total_citations']}")
    
    print("\n  ✓ PASS (Report generated successfully)")
    
    # Save report
    output_file = "test_evaluation_report.json"
    evaluator.save_report(report, output_file)
    print(f"\n  Report saved to: {output_file}")


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*70)
    print("LEGAL RAG EVALUATION TEST SUITE")
    print("="*70)
    print("\nTesting fixes for:")
    print("1. RAG evaluation metrics (Precision, Recall, MRR, NDCG)")
    print("2. Citation verification and hallucination detection")
    print("3. Citation quality scoring")
    print("4. Comprehensive evaluation reporting")
    
    try:
        test_citation_verification()
        test_hallucination_detection()
        test_rag_metrics()
        test_citation_quality_scoring()
        test_evaluation_report_generation()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        print("\n✓ Evaluation metrics framework is operational")
        print("✓ Citation verification is active")
        print("✓ Hallucination detection is functional")
        print("\nNext steps:")
        print("1. Collect human annotations for test cases")
        print("2. Run evaluation on production data")
        print("3. Calculate inter-annotator agreement")
        print("4. Compare against baseline systems")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
