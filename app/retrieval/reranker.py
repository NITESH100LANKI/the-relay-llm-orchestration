"""
Hybrid RRF (Reciprocal Rank Fusion) and Cross-Scoring Reranker for The Relay.
Merges candidate passages from BM25 (lexical) and Dense (semantic) retrievers.
"""

from typing import List, Dict
from app.schemas.models import RetrievedPassage

def reciprocal_rank_fusion(
    bm25_results: List[RetrievedPassage],
    dense_results: List[RetrievedPassage],
    rrf_k: int = 60,
    top_k: int = 5
) -> List[RetrievedPassage]:
    """
    Combines BM25 and Dense results using standard Reciprocal Rank Fusion:
    score(doc) = 1 / (k + rank_bm25) + 1 / (k + rank_dense)
    """
    scores: Dict[str, float] = {}
    passage_map: Dict[str, RetrievedPassage] = {}

    for rank, p in enumerate(bm25_results, start=1):
        passage_map[p.chunk_id] = p
        scores[p.chunk_id] = scores.get(p.chunk_id, 0.0) + (1.0 / (rrf_k + rank))

    for rank, p in enumerate(dense_results, start=1):
        if p.chunk_id not in passage_map:
            passage_map[p.chunk_id] = p
        scores[p.chunk_id] = scores.get(p.chunk_id, 0.0) + (1.0 / (rrf_k + rank))

    # Sort descending by fused score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    reranked: List[RetrievedPassage] = []
    for rank, (chunk_id, score) in enumerate(sorted_items[:top_k], start=1):
        original = passage_map[chunk_id]
        reranked.append(
            RetrievedPassage(
                chunk_id=original.chunk_id,
                paper_title=original.paper_title,
                arxiv_id=original.arxiv_id,
                version=original.version,
                page_number=original.page_number,
                section=original.section,
                text=original.text,
                score=score,
                rank=rank
            )
        )
    return reranked

