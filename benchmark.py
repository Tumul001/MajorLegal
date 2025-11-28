"""
Automated Benchmarking for Legal RAG System

Evaluates the RAG system's performance against baselines:
1. Vanilla LLM (Gemini/GPT-4) - Batched
2. Simple RAG (Vector only, no graph/validation)
3. MajorLegal (Full system: Vector + Graph + Validation)
"""

import json
import time
import os
from typing import List, Dict
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Import LLM for Vanilla Baseline
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ langchain_google_genai not found. Vanilla baseline will be skipped.")

class LegalBenchmarkSuite:
    """Comprehensive Legal Benchmark Suite"""
    
    def __init__(self, skip_vanilla: bool = False):
        self.results = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file = "bench_result.txt"
        self.skip_vanilla = skip_vanilla
        
        # Initialize RAG System
        try:
            from rag_system.legal_rag import ProductionLegalRAGSystem
            self.rag_system = ProductionLegalRAGSystem()
        except Exception as e:
            print(f"⚠️ Could not initialize RAG system: {e}")
            self.rag_system = None

        # Initialize LLM for Vanilla Baseline
        self.llm = None
        if not self.skip_vanilla and LLM_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            # Use gemini-2.0-flash for speed and availability
            # Set max_retries to 1 to avoid hanging on rate limits
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, max_retries=1)
        elif not self.skip_vanilla:
            print("⚠️ Vanilla LLM skipped due to missing API key or library.")
        
    def load_questions(self, filepath: str = "data/benchmark_questions.json") -> List[Dict]:
        """Load questions from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _fuzzy_match(self, str1: str, str2: str) -> bool:
        """Check if two strings match using token overlap"""
        s1 = set(str1.lower().replace('.', '').replace(',', '').split())
        s2 = set(str2.lower().replace('.', '').replace(',', '').split())
        min_len = min(len(s1), len(s2))
        if min_len == 0: return False
        return len(s1.intersection(s2)) / min_len >= 0.5

    def calculate_metrics(self, retrieved_citations: List[str], expected_citations: List[str]) -> Dict:
        """Calculate Precision, Recall, F1"""
        if not expected_citations:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        # Recall
        matches_recall = 0
        for expected in expected_citations:
            for retrieved in retrieved_citations:
                if self._fuzzy_match(retrieved, expected):
                    matches_recall += 1
                    break
        recall = matches_recall / len(expected_citations)
        
        # Precision
        matches_precision = 0
        if not retrieved_citations:
            precision = 0.0
        else:
            for retrieved in retrieved_citations:
                for expected in expected_citations:
                    if self._fuzzy_match(retrieved, expected):
                        matches_precision += 1
                        break
            precision = matches_precision / len(retrieved_citations)
            
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}

    def run_vanilla_llm_batch(self, questions: List[Dict], batch_size: int = 40) -> List[Dict]:
        """Run Vanilla LLM baseline in batches"""
        if not self.llm:
            return [{"error": "LLM not initialized"}] * len(questions)
            
        print(f"\n🤖 Running Vanilla LLM Baseline ({len(questions)} questions, batch size {batch_size})...")
        results = []
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            print(f"   Processing batch {i//batch_size + 1}...")
            
            for q in tqdm(batch, desc="Vanilla Batch"):
                try:
                    prompt = f"Question: {q['question']}\n\nAnswer the question and list relevant Indian case laws and statutes as citations."
                    response = self.llm.invoke(prompt)
                    content = response.content
                    
                    # Extract citations (naive extraction for baseline)
                    # Assuming citations are mentioned in text. We'll treat the whole answer as context 
                    # but for metrics we need a list. We'll split by newlines or look for "v."
                    # For fair comparison, we'll just check if expected citations appear in the text.
                    
                    # Actually, to be fair with RAG which returns a list, we should ask LLM to output a list.
                    # But the user said "Ask Gemini/GPT-4 the same questions".
                    # We will check if expected citations are present in the response text.
                    
                    found_citations = []
                    # This is a bit generous to Vanilla, but fair enough.
                    # We will simulate "retrieved citations" by checking if expected ones are in the text.
                    # Wait, that makes Recall 1.0 if it mentions them, but Precision?
                    # To calculate precision, we need to know what *else* it cited.
                    # Let's ask for a specific format.
                    
                    prompt_fmt = f"""Question: {q['question']}
                    
                    Provide a legal answer. Then, list relevant citations in a separate section titled "Citations:".
                    """
                    response = self.llm.invoke(prompt_fmt)
                    content = response.content
                    
                    # Parse citations
                    retrieved = []
                    # Robust extraction using regex
                    import re
                    match = re.search(r'(?i)(?:^|\n)\s*(?:#+|[*]+)?\s*Citations\s*(?:[:]+)?\s*(?:[*]+)?\s*(?:\n|$)', content)
                    if match:
                        citation_section = content[match.end():]
                        # Stop at next header if any
                        next_header = re.search(r'(?i)(?:^|\n)\s*(?:#+|[*]+)?\s*(?:Conclusion|Reasoning|Explanation)', citation_section)
                        if next_header:
                            citation_section = citation_section[:next_header.start()]
                            
                        retrieved = [line.strip().strip('-*•1234567890. ') for line in citation_section.split('\n') if line.strip()]
                    
                    metrics = self.calculate_metrics(retrieved, q['expected_citations'])
                    results.append({
                        "system": "Vanilla LLM",
                        "question_id": q.get('id'),
                        "metrics": metrics
                    })
                    
                except Exception as e:
                    print(f"Error in Vanilla LLM: {e}")
                    results.append({"system": "Vanilla LLM", "metrics": {"precision": 0, "recall": 0, "f1": 0}})
                
                time.sleep(1) # Rate limit safety
                
        return results

    def run_simple_rag(self, question: Dict) -> Dict:
        """Run Simple RAG (Vector only, no validation)"""
        if not self.rag_system: return {}
        
        # Force vector only, no validation
        docs = self.rag_system.retrieve_documents(question['question'], k=3, validate=False)
        
        retrieved_citations = []
        for doc in docs:
            if doc.metadata.get('case_name'): retrieved_citations.append(doc.metadata['case_name'])
            if doc.metadata.get('citation') and doc.metadata['citation'] != 'N/A': retrieved_citations.append(doc.metadata['citation'])
            
        metrics = self.calculate_metrics(retrieved_citations, question['expected_citations'])
        return {"system": "Simple RAG", "question_id": question.get('id'), "metrics": metrics}

    def run_major_legal(self, question: Dict) -> Dict:
        """Run MajorLegal (Full System)"""
        if not self.rag_system: return {}
        
        # Full system: Graph-RAG + Validation
        # Note: retrieve_documents uses self.use_graph_rag internally which defaults to True if initialized
        docs = self.rag_system.retrieve_documents(question['question'], k=3, validate=True)
        
        retrieved_citations = []
        for doc in docs:
            if doc.metadata.get('case_name'): retrieved_citations.append(doc.metadata['case_name'])
            if doc.metadata.get('citation') and doc.metadata['citation'] != 'N/A': retrieved_citations.append(doc.metadata['citation'])
            
        metrics = self.calculate_metrics(retrieved_citations, question['expected_citations'])
        return {"system": "MajorLegal", "question_id": question.get('id'), "metrics": metrics}

    def run_benchmark(self, limit: int = None):
        print(f"🚀 Starting Comprehensive Benchmark at {self.timestamp}")
        questions = self.load_questions()
        if limit:
            questions = questions[:limit]
            print(f"⚠️ Limiting to first {limit} questions for testing.")
            
        print(f"📝 Loaded {len(questions)} questions.")
        
        all_results = []
        
        # 1. Run Vanilla LLM (Batched)
        if not self.skip_vanilla and self.llm:
            vanilla_results = self.run_vanilla_llm_batch(questions)
            all_results.extend(vanilla_results)
        else:
            print("\n⏭️ Skipping Vanilla LLM Baseline.")
        
        # 2. Run RAG Systems
        print("\n⚙️ Running RAG Baselines...")
        for q in tqdm(questions, desc="RAG Evaluation"):
            # Simple RAG
            simple_res = self.run_simple_rag(q)
            all_results.append(simple_res)
            
            # MajorLegal
            major_res = self.run_major_legal(q)
            all_results.append(major_res)
            
        self.log_results(all_results)
        self.print_summary(all_results)

    def log_results(self, results: List[Dict]):
        """Log results to bench_result.txt with timestamp"""
        
        # Calculate averages
        systems = ["Vanilla LLM", "Simple RAG", "MajorLegal"]
        summary_lines = []
        summary_lines.append(f"\n{'='*50}")
        summary_lines.append(f"BENCHMARK RUN: {self.timestamp}")
        summary_lines.append(f"{'='*50}")
        
        for sys_name in systems:
            sys_res = [r for r in results if r.get('system') == sys_name]
            if not sys_res: continue
            
            avg_p = sum(r['metrics']['precision'] for r in sys_res) / len(sys_res)
            avg_r = sum(r['metrics']['recall'] for r in sys_res) / len(sys_res)
            avg_f1 = sum(r['metrics']['f1'] for r in sys_res) / len(sys_res)
            
            summary_lines.append(f"\nSystem: {sys_name}")
            summary_lines.append(f"  Precision: {avg_p:.4f}")
            summary_lines.append(f"  Recall:    {avg_r:.4f}")
            summary_lines.append(f"  F1 Score:  {avg_f1:.4f}")
            
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
            
        print(f"\n✅ Results logged to {self.log_file}")

    def print_summary(self, results: List[Dict]):
        # Same as log but print to console
        systems = ["Vanilla LLM", "Simple RAG", "MajorLegal"]
        print(f"\n{'='*50}")
        print(f"📊 BENCHMARK SUMMARY")
        print(f"{'='*50}")
        
        for sys_name in systems:
            sys_res = [r for r in results if r.get('system') == sys_name]
            if not sys_res: continue
            
            avg_p = sum(r['metrics']['precision'] for r in sys_res) / len(sys_res)
            avg_r = sum(r['metrics']['recall'] for r in sys_res) / len(sys_res)
            avg_f1 = sum(r['metrics']['f1'] for r in sys_res) / len(sys_res)
            
            print(f"\nSystem: {sys_name}")
            print(f"  Precision: {avg_p:.4f}")
            print(f"  Recall:    {avg_r:.4f}")
            print(f"  F1 Score:  {avg_f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Legal RAG Benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--skip-vanilla", action="store_true", help="Skip Vanilla LLM baseline")
    args = parser.parse_args()

    suite = LegalBenchmarkSuite(skip_vanilla=args.skip_vanilla)
    suite.run_benchmark(limit=args.limit)
