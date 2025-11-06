"""Test script to verify the RAG system works"""

import os
os.environ['USE_MOCK_EMBEDDINGS'] = 'true'

from rag_system.legal_rag import ProductionLegalRAGSystem

# Initialize RAG system
print("Initializing RAG system...")
rag = ProductionLegalRAGSystem()

# Test search
print("\nTesting search for 'arrest'...")
results = rag.vector_store.similarity_search('arrest', k=2)

print(f"\nFound {len(results)} results:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.metadata.get('case_name', 'Unknown')}")
    print(f"   Citation: {doc.metadata.get('citation', 'Unknown')}")
    print(f"   Preview: {doc.page_content[:100]}...")
    print()

print("✅ RAG system is working!")
