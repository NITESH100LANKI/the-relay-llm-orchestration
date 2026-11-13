"""
BM25 Lexical Retriever for The Relay.
Uses rank_bm25.BM25Okapi to retrieve relevant document chunks based on exact
lexical matching, technical terms, and keyword frequencies.
"""

import re
import string
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from app.schemas.models import DocumentChunk, RetrievedPassage

def tokenize_for_bm25(text: str) -> List[str]:
    """
    Standard lowercased word tokenization with basic punctuation stripping.
    Preserves hyphenated technical terms (e.g. multi-head, cross-attention).
    """
    # Lowercase
    text = text.lower()
    # Match words and hyphenated words
    tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text)
    return tokens

class BM25Retriever:
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.corpus = [tokenize_for_bm25(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[RetrievedPassage]:
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []
            
        scores = self.bm25.get_scores(query_tokens)
        
        # Pair chunk with score and sort descending
        scored_pairs = []
        for idx, score in enumerate(scores):
            if score >= min_score and score > 0:
                scored_pairs.append((self.chunks[idx], float(score)))
                
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = scored_pairs[:top_k]
        
        results: List[RetrievedPassage] = []
        for rank, (chunk, score) in enumerate(top_pairs, start=1):
            results.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    paper_title=chunk.paper_title,
                    arxiv_id=chunk.arxiv_id,
                    version=chunk.version,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    text=chunk.text,
                    score=score,
                    rank=rank
                )
            )
        return results

