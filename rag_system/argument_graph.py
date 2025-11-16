"""
rag_system/argument_graph.py

Build an explainable argument graph from prosecution and defense arguments.

This module extracts atomic claims (naively via sentence splitting), creates nodes
for main arguments, supporting points, citations, statutes, and weaknesses, and
connects them with edges describing relations (supports, cites, acknowledges).

The graph is serializable to JSON using networkx.node_link_data.
"""
from __future__ import annotations

from typing import List, Dict, Any
import networkx as nx
import os
import json
from pathlib import Path


def _safe_text(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def extract_atomic_claims(argument: Any) -> List[Dict[str, Any]]:
    """Extract a list of atomic claims from a LegalArgument-like object.

    This is a lightweight heuristic: split main_argument and legal_reasoning
    into sentence-like pieces and combine supporting_points as distinct claims.
    """
    claims = []
    main = _safe_text(getattr(argument, 'main_argument', ''))
    if main:
        claims.append({'type': 'main', 'text': main})

    for i, pt in enumerate(getattr(argument, 'supporting_points', []) or []):
        text = _safe_text(pt)
        if text:
            claims.append({'type': 'supporting_point', 'text': text, 'index': i})

    reasoning = _safe_text(getattr(argument, 'legal_reasoning', ''))
    # Naive sentence split on periods
    for i, sent in enumerate([s.strip() for s in reasoning.split('.') if s.strip()]):
        claims.append({'type': 'reasoning', 'text': sent, 'index': i})

    # Weaknesses as negative claims
    for i, w in enumerate(getattr(argument, 'weaknesses_acknowledged', []) or []):
        text = _safe_text(w)
        if text:
            claims.append({'type': 'weakness', 'text': text, 'index': i})

    return claims


def build_argument_graph(prosecution_args: List[Any], defense_args: List[Any]) -> Dict:
    """Build a directed argument graph from lists of LegalArgument-like objects.

    Returns a JSON-serializable dict (node-link format).
    Node types: main, supporting_point, reasoning, weakness, case, statute
    Edge attributes: relation in {'supports','cites','acknowledges','uses_statute','rebuts'}
    """
    G = nx.DiGraph()

    def _add_argument_nodes(role: str, args: List[Any]):
        for r_idx, arg in enumerate(args, start=1):
            main_id = f"{role}_round{r_idx}_main"
            main_text = _safe_text(getattr(arg, 'main_argument', ''))
            G.add_node(main_id, type='main', role=role, round=r_idx, text=main_text)

            # Supporting points
            for i, sp in enumerate(getattr(arg, 'supporting_points', []) or []):
                sp_id = f"{role}_round{r_idx}_sp{i+1}"
                G.add_node(sp_id, type='supporting_point', text=_safe_text(sp), role=role, round=r_idx)
                G.add_edge(main_id, sp_id, relation='supports')

            # Reasoning sentences
            reasoning = _safe_text(getattr(arg, 'legal_reasoning', ''))
            for i, sent in enumerate([s.strip() for s in reasoning.split('.') if s.strip()]):
                rs_id = f"{role}_round{r_idx}_rs{i+1}"
                G.add_node(rs_id, type='reasoning', text=sent, role=role, round=r_idx)
                G.add_edge(main_id, rs_id, relation='supports')

            # Weaknesses
            for i, w in enumerate(getattr(arg, 'weaknesses_acknowledged', []) or []):
                w_id = f"{role}_round{r_idx}_weak{i+1}"
                G.add_node(w_id, type='weakness', text=_safe_text(w), role=role, round=r_idx)
                G.add_edge(main_id, w_id, relation='acknowledges')

            # Case citations
            for j, citation in enumerate(getattr(arg, 'case_citations', []) or []):
                case_name = _safe_text(getattr(citation, 'case_name', citation.get('case_name') if isinstance(citation, dict) else ''))
                citation_id = f"case_{(case_name or 'unknown').replace(' ', '_')}_{j+1}"
                # store metadata where available
                meta = {}
                if isinstance(citation, dict):
                    meta = citation
                else:
                    # try attribute access
                    for k in ['case_name', 'citation', 'year', 'relevance', 'excerpt']:
                        meta[k] = _safe_text(getattr(citation, k, ''))

                if citation_id not in G:
                    G.add_node(citation_id, type='case', text=meta.get('case_name', ''), metadata=meta)

                # Connect main and supporting points to the case node
                G.add_edge(main_id, citation_id, relation='cites')
                for i, sp in enumerate(getattr(arg, 'supporting_points', []) or []):
                    sp_id = f"{role}_round{r_idx}_sp{i+1}"
                    if sp_id in G:
                        G.add_edge(sp_id, citation_id, relation='cites')

            # Statutes
            for k, statute in enumerate(getattr(arg, 'statutes_cited', []) or []):
                st_id = f"statute_{str(statute).replace(' ', '_')}_{k+1}"
                if st_id not in G:
                    G.add_node(st_id, type='statute', text=_safe_text(statute))
                G.add_edge(main_id, st_id, relation='uses_statute')

    _add_argument_nodes('prosecution', prosecution_args)
    _add_argument_nodes('defense', defense_args)

    # Simple rebuttal links: if a defense supporting point contains words like 'but','however','however,' link as rebuts
    for node_id, data in list(G.nodes(data=True)):
        if data.get('type') == 'supporting_point' and data.get('role') == 'defense':
            text = data.get('text','').lower()
            if any(tok in text for tok in ('but', 'however', 'although', 'despite')):
                # find prosecution main nodes that share keywords and link
                for p_node, p_data in G.nodes(data=True):
                    if p_data.get('type') == 'main' and p_data.get('role') == 'prosecution':
                        # naive keyword overlap
                        if any(w in p_data.get('text','').lower() for w in text.split()[:6]):
                            G.add_edge(node_id, p_node, relation='rebuts')

    # Return node-link JSON serializable dict
    data = nx.node_link_data(G)
    return data


def save_graph_json(graph_data: Dict, out_dir: str, filename: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    return str(path)
