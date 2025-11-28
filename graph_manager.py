"""
Citation Graph Manager for Legal RAG System

Implements Graph-RAG by building a citation network and using PageRank
to identify influential cases for hybrid retrieval.
"""

import networkx as nx
import re
from typing import List, Dict, Tuple
import pickle
from pathlib import Path
import json


class CitationGraph:
    """Manages citation network analysis for legal cases"""
    
    def __init__(self, graph_path: str = "data/citation_graph.pkl"):
        """
        Initialize citation graph
        
        Args:
            graph_path: Path to save/load the citation graph
        """
        self.graph = nx.DiGraph()
        self.graph_path = Path(graph_path)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.pagerank_scores = {}
    
    def extract_citations(self, case_text: str) -> List[str]:
        """
        Extract case citations from text using regex heuristics
        
        Patterns matched:
        - "v." or "vs." (versus in case names)
        - "AIR" (All India Reporter) citations
        - "SCC" (Supreme Court Cases) citations
        - Common citation formats
        
        Args:
            case_text: Full text of the legal case
        
        Returns:
            List of extracted citation strings
        """
        citations = []
        
        # Pattern 1: Case name with v. or vs.
        # Example: "Ram v. Shyam", "State vs. Accused"
        case_name_pattern = r'([A-Z][a-zA-Z\s&,\.]+)\s+(?:v\.|vs\.)\s+([A-Z][a-zA-Z\s&,\.]+)'
        matches = re.findall(case_name_pattern, case_text)
        for match in matches:
            citation = f"{match[0].strip()} v. {match[1].strip()}"
            if len(citation) < 100:  # Reasonable case name length
                citations.append(citation)
        
        # Pattern 2: AIR citations
        # Example: "AIR 1978 SC 597", "1978 AIR 597"
        air_pattern = r'(?:AIR\s+)?(\d{4})\s+(?:AIR\s+)?([A-Z]{2,})\s+(\d+)'
        air_matches = re.findall(air_pattern, case_text)
        citations.extend([f"{m[0]} {m[1]} {m[2]}" for m in air_matches])
        
        # Pattern 3: SCC citations
        # Example: "(1978) 1 SCC 248"
        scc_pattern = r'\((\d{4})\)\s+(\d+)\s+SCC\s+(\d+)'
        scc_matches = re.findall(scc_pattern, case_text)
        citations.extend([f"({m[0]}) {m[1]} SCC {m[2]}" for m in scc_matches])
        
        return list(set(citations))  # Remove duplicates
    
    def build_citation_graph(self, cases: List[Dict]) -> nx.DiGraph:
        """
        Build citation network from case documents
        
        Args:
            cases: List of case dictionaries with 'case_name' and 'text' keys
        
        Returns:
            NetworkX directed graph with citation edges
        """
        print(f"🕸️  Building citation graph from {len(cases)} cases...")
        
        # Add all cases as nodes first
        for case in cases:
            case_name = case.get('case_name', case.get('metadata', {}).get('case_name', 'Unknown'))
            self.graph.add_node(
                case_name,
                citation=case.get('citation', case.get('metadata', {}).get('citation', '')),
                court=case.get('court', case.get('metadata', {}).get('court', ''))
            )
        
        # Extract citations and create edges
        edge_count = 0
        for case in cases:
            citing_case = case.get('case_name', case.get('metadata', {}).get('case_name', 'Unknown'))
            case_text = case.get('text', '')
            
            # Extract citations from this case
            cited_cases = self.extract_citations(case_text)
            
            # Create edges for each citation
            for cited_case in cited_cases:
                # Only create edge if cited case exists in our dataset
                if cited_case in self.graph.nodes:
                    self.graph.add_edge(citing_case, cited_case)
                    edge_count += 1
        
        print(f"✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
    
    def compute_pagerank(self, alpha: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        """
        Compute PageRank scores for all cases in the network
        
        Args:
            alpha: Damping parameter (default: 0.85)
            max_iter: Maximum iterations
        
        Returns:
            Dictionary mapping case names to PageRank scores (0-1)
        """
        if self.graph.number_of_nodes() == 0:
            print("⚠️  Empty graph - cannot compute PageRank")
            return {}
        
        print("📊 Computing PageRank scores...")
        
        try:
            self.pagerank_scores = nx.pagerank(
                self.graph,
                alpha=alpha,
                max_iter=max_iter
            )
            
            # Normalize to 0-1 range
            max_score = max(self.pagerank_scores.values()) if self.pagerank_scores else 1.0
            if max_score > 0:
                self.pagerank_scores = {
k: v / max_score
                    for k, v in self.pagerank_scores.items()
                }
            
            print(f"✅ PageRank computed for {len(self.pagerank_scores)} cases")
            
            # Show top 5 most influential cases
            top_cases = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\n📈 Top 5 Most Influential Cases:")
            for idx, (case, score) in enumerate(top_cases, 1):
                print(f"   {idx}. {case[:60]}... (score: {score:.4f})")
            
            return self.pagerank_scores
            
        except Exception as e:
            print(f"❌ PageRank computation failed: {e}")
            return {}
    
    def get_pagerank_score(self, case_name: str) -> float:
        """
        Get PageRank score for a specific case
        
        Args:
            case_name: Name of the case
        
        Returns:
            PageRank score (0-1), or 0.0 if case not found
        """
        return self.pagerank_scores.get(case_name, 0.0)
    
    def save_graph(self):
        """Save citation graph to disk"""
        with open(self.graph_path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'pagerank_scores': self.pagerank_scores
            }, f)
        print(f"💾 Citation graph saved to {self.graph_path}")
    
    def load_graph(self):
        """Load citation graph from disk"""
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph not found at {self.graph_path}")
        
        with open(self.graph_path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.pagerank_scores = data['pagerank_scores']
        
        print(f"✅ Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def visualize_graph(self, output_path: str = "data/citation_graph.graphml"):
        """
        Export graph to GraphML format for visualization
        
        Can be opened in tools like Gephi or Cytoscape
        """
        nx.write_graphml(self.graph, output_path)
        print(f"🎨 Graph exported to {output_path}")
    
    def get_stats(self) -> Dict:
        """Get citation graph statistics"""
        if self.graph.number_of_nodes() == 0:
            return {"status": "Empty graph"}
        
        return {
            "total_cases": self.graph.number_of_nodes(),
            "total_citations": self.graph.number_of_edges(),
            "avg_citations_per_case": self.graph.number_of_edges() / self.graph.number_of_nodes(),
            "most_cited_case": max(self.pagerank_scores, key=self.pagerank_scores.get) if self.pagerank_scores else "N/A"
        }


def build_graph_from_processed_data(processed_data_path: str = "data/processed/ildc_cases.json"):
    """
    Build citation graph from processed ILDC data
    
    Args:
        processed_data_path: Path to processed case data
    
    Returns:
        CitationGraph object
    """
    print("=" * 60)
    print("🚀 Building Citation Graph")
    print("=" * 60)
    
    # Load processed data
    with open(processed_data_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    # Build graph
    citation_graph = CitationGraph()
    citation_graph.build_citation_graph(cases)
    
    # Compute PageRank
    citation_graph.compute_pagerank()
    
    # Save graph
    citation_graph.save_graph()
    
    # Print stats
    stats = citation_graph.get_stats()
    print("\n📊 Citation Graph Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("=" * 60)
    return citation_graph


if __name__ == "__main__":
    # Build graph from processed data
    graph = build_graph_from_processed_data()
    
    # Optional: visualize
    graph.visualize_graph()
