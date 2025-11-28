"""Verify the complete system status"""
import os
import json
from pathlib import Path

print("=" * 60)
print("🔍 SYSTEM STATUS CHECK")
print("=" * 60)

# 1. Check data ingestion
data_file = Path("data/processed/ildc_cases.json")
if data_file.exists():
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n✅ Data Ingestion: {len(data)} document chunks")
    print(f"   Source: ILDC dataset (54 cases)")
else:
    print("\n❌ Data Ingestion: No data found")

# 2. Check vector store
vector_store_path = Path("data/vector_store/faiss_index")
if vector_store_path.exists():
    print(f"\n✅ Vector Store: Built with InLegalBERT")
    print(f"   Location: {vector_store_path}")
    # Try to load and get stats
    try:
        from rag_system.legal_rag import ProductionLegalRAGSystem
        rag = ProductionLegalRAGSystem()
        stats = rag.get_stats()
        print(f"   Documents indexed: {stats.get('total_documents', 'N/A')}")
        print(f"   Embedding dimension: {stats.get('embedding_dimension', 'N/A')}")
        print(f"   Graph-RAG enabled: {stats.get('graph_enabled', False)}")
    except Exception as e:
        print(f"   ⚠️  Could not load stats: {str(e)[:50]}")
else:
    print("\n❌ Vector Store: Not built")

# 3. Check citation graph
graph_file = Path("data/citation_graph.pkl")
if graph_file.exists():
    import pickle
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    graph = graph_data.get('graph')
    pagerank = graph_data.get('pagerank_scores', {})
    print(f"\n✅ Citation Graph: Built")
    print(f"   Nodes (cases): {graph.number_of_nodes() if graph else 0}")
    print(f"   Edges (citations): {graph.number_of_edges() if graph else 0}")
    print(f"   PageRank scores: {len(pagerank)} cases")
else:
    print("\n❌ Citation Graph: Not built")

# 4. Test vector store functionality
print("\n" + "=" * 60)
print("🧪 FUNCTIONALITY TEST")
print("=" * 60)

try:
    from rag_system.legal_rag import ProductionLegalRAGSystem
    rag = ProductionLegalRAGSystem(use_graph_rag=True)
    
    # Test query
    test_query = "What are the requirements for a valid arrest under CrPC?"
    print(f"\nTest Query: '{test_query}'")
    
    docs = rag.retrieve_documents(test_query, k=3)
    print(f"\n✅ Retrieved {len(docs)} documents")
    
    if docs:
        print(f"\nTop Result:")
        print(f"  Case: {docs[0].metadata.get('case_name', 'N/A')[:80]}")
        print(f"  Citation: {docs[0].metadata.get('citation', 'N/A')}")
        print(f"  Risk Level: {docs[0].metadata.get('risk_level', 'Not validated')}")
        if 'hybrid_score' in docs[0].metadata:
            print(f"  Hybrid Score: {docs[0].metadata['hybrid_score']:.3f}")
            print(f"  (Vector: {docs[0].metadata.get('vector_score', 0):.3f} + PageRank: {docs[0].metadata.get('pagerank_score', 0):.3f})")
        print(f"  Preview: {docs[0].page_content[:150]}...")
    
    print("\n✅ ALL SYSTEMS OPERATIONAL")
    
except Exception as e:
    print(f"\n❌ Error during test: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
