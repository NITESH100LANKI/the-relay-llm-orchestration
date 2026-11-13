"""
Retrieval Engine for The Relay.
Provides the authoritative entry point:
    retrieve_paper_passages(question, retrieval_config)
Returns ranked passages with complete source metadata.
Supports Configurations A, B, and C.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union

from app.schemas.models import (
    DocumentPage,
    DocumentChunk,
    RetrievalConfig,
    RetrievedPassage
)
from app.retrieval.chunker import chunk_pages
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.reranker import reciprocal_rank_fusion
from app.core.config import settings

CONFIG_PRESETS: Dict[str, RetrievalConfig] = {
    "A": RetrievalConfig(
        name="A",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
        rerank_enabled=False,
        retrieval_mode="hybrid"
    ),
    "B": RetrievalConfig(
        name="B",
        chunk_size=500,
        chunk_overlap=100,
        top_k=5,
        rerank_enabled=False,
        retrieval_mode="hybrid"
    ),
    "C": RetrievalConfig(
        name="C",
        chunk_size=500,
        chunk_overlap=100,
        top_k=5,
        rerank_enabled=True,
        retrieval_mode="hybrid"
    ),
}

class RetrievalIndex:
    def __init__(self, config: RetrievalConfig, pages: List[DocumentPage]):
        self.config = config
        self.pages = pages
        self.chunks = chunk_pages(pages, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        print(f"[RetrievalIndex:{config.name}] Built {len(self.chunks)} chunks (size={config.chunk_size}, overlap={config.chunk_overlap}).")
        
        self.bm25 = BM25Retriever(self.chunks)
        self.dense = DenseRetriever(self.chunks, cache_dir=settings.indexes_dir)

    def retrieve(self, question: str) -> List[RetrievedPassage]:
        top_k = self.config.top_k
        if not self.config.rerank_enabled:
            # Single or primary retrieval mode
            bm25_hits = self.bm25.retrieve(question, top_k=top_k)
            dense_hits = self.dense.retrieve(question, top_k=top_k)
            # If reranking is disabled, use best dense or bm25 depending on query length
            if len(question.split()) <= 4:
                # Short keyword query: bm25 performs reliably
                return bm25_hits if bm25_hits else dense_hits
            else:
                return dense_hits if dense_hits else bm25_hits
        else:
            # Hybrid with Reciprocal Rank Fusion
            candidate_k = max(10, top_k * 2)
            bm25_hits = self.bm25.retrieve(question, top_k=candidate_k)
            dense_hits = self.dense.retrieve(question, top_k=candidate_k)
            return reciprocal_rank_fusion(bm25_hits, dense_hits, rrf_k=60, top_k=top_k)

_LOADED_PAGES: Optional[List[DocumentPage]] = None
_INDEX_CACHE: Dict[str, RetrievalIndex] = {}

def load_pages() -> List[DocumentPage]:
    global _LOADED_PAGES
    if _LOADED_PAGES is not None:
        return _LOADED_PAGES

    extracted_path = settings.processed_dir / "extracted_pages.json"
    if not extracted_path.exists():
        from app.retrieval.extractor import extract_corpus
        _LOADED_PAGES = extract_corpus(settings.corpus_dir / "manifest.json", settings.processed_dir)
    else:
        with open(extracted_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _LOADED_PAGES = [DocumentPage(**item) for item in data]
    return _LOADED_PAGES

def get_retrieval_index(config: RetrievalConfig) -> RetrievalIndex:
    cache_key = f"{config.name}_{config.chunk_size}_{config.chunk_overlap}_{config.rerank_enabled}"
    if cache_key not in _INDEX_CACHE:
        pages = load_pages()
        _INDEX_CACHE[cache_key] = RetrievalIndex(config, pages)
    return _INDEX_CACHE[cache_key]

def retrieve_paper_passages(
    question: str,
    retrieval_config: Union[RetrievalConfig, str]
) -> List[RetrievedPassage]:
    """
    Primary API function:
    retrieve_paper_passages(question, retrieval_config)
    Returns ranked passages with complete source metadata.
    """
    if isinstance(retrieval_config, str):
        cfg = CONFIG_PRESETS.get(retrieval_config.upper(), CONFIG_PRESETS["B"])
    else:
        cfg = retrieval_config
        
    index = get_retrieval_index(cfg)
    return index.retrieve(question)

