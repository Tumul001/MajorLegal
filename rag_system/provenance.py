"""
rag_system/provenance.py

Simple provenance linker: for each supporting point and reasoning sentence in a
LegalArgument, run a similarity search against the vector store and attach the
top-k retrieved documents (doc id, title, score, excerpt).

This is intentionally lightweight; it creates structured evidence items that
can be displayed and saved with each argument for explainability and evaluation.
"""
from typing import Any, Dict, List
from pathlib import Path
import json


def link_claims_provenance(argument: Any, vector_store, top_k: int = 3) -> List[Dict]:
    """Return a provenance list for the given argument.

    Each provenance item maps a claim text -> list of retrieved documents with score.
    Document entries include: case_name, citation, score, excerpt, doc_id (if available).
    """
    provenance = []

    # Collect claim texts: main, supporting points, reasoning sentences
    claims = []
    main = getattr(argument, 'main_argument', None)
    if main:
        claims.append({'role': 'main', 'text': str(main)})

    for sp in getattr(argument, 'supporting_points', []) or []:
        claims.append({'role': 'supporting_point', 'text': str(sp)})

    reasoning = getattr(argument, 'legal_reasoning', '') or ''
    # naive split
    for sent in [s.strip() for s in reasoning.split('.') if s.strip()]:
        claims.append({'role': 'reasoning', 'text': sent})

    # For each claim, run vector search
    for claim in claims:
        text = claim['text']
        try:
            results = vector_store.similarity_search_with_score(text, k=top_k)
        except Exception:
            # Fallback to similarity_search (some vectorstores may not support scores)
            try:
                docs = vector_store.similarity_search(text, k=top_k)
                results = [(d, None) for d in docs]
            except Exception:
                results = []

        mapped = []
        for doc, score in results:
            meta = doc.metadata if hasattr(doc, 'metadata') else getattr(doc, 'extra', {})
            mapped.append({
                'case_name': meta.get('case_name') if isinstance(meta, dict) else None,
                'citation': meta.get('citation') if isinstance(meta, dict) else None,
                'doc_id': meta.get('url') or meta.get('id') or meta.get('chunk_index'),
                'score': float(score) if score is not None else None,
                'excerpt': (doc.page_content[:300] + '...') if getattr(doc, 'page_content', None) else ''
            })

        provenance.append({
            'claim_text': text,
            'role': claim.get('role'),
            'evidence': mapped
        })

    # Attach provenance to argument if possible
    try:
        setattr(argument, 'provenance', provenance)
    except Exception:
        # Best-effort: ignore
        pass

    return provenance
