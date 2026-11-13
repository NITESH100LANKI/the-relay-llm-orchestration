"""
Dense Semantic Retriever for The Relay.
Uses HuggingFace Transformers (AutoTokenizer, AutoModel) with mean pooling and L2 normalization
to compute semantic embeddings locally, with disk caching.
Also includes a zero-dependency pure-Python TF-IDF fallback.
"""

import os
import math
import collections
from pathlib import Path
from typing import List, Optional
import numpy as np

from app.schemas.models import DocumentChunk, RetrievedPassage

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging.
    """
    import torch
    token_embeddings = model_output[0]  # First element contains hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class PurePythonTFIDF:
    """Zero-dependency pure-Python TF-IDF implementation as fallback."""
    def __init__(self, documents: List[str]):
        self.doc_tokens = [self._tokenize(doc) for doc in documents]
        self.num_docs = len(self.doc_tokens)
        self.df = collections.defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1
        self.idf = {term: math.log((self.num_docs + 1) / (count + 1)) + 1.0 for term, count in self.df.items()}
        self.doc_vectors = [self._vectorize(tokens) for tokens in self.doc_tokens]

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def _vectorize(self, tokens: List[str]) -> dict:
        counts = collections.Counter(tokens)
        vec = {}
        norm_sq = 0.0
        for term, count in counts.items():
            if term in self.idf:
                tf = count / len(tokens)
                w = tf * self.idf[term]
                vec[term] = w
                norm_sq += w * w
        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        return {term: w / norm for term, w in vec.items()}

    def score(self, query: str) -> List[float]:
        q_tokens = self._tokenize(query)
        q_vec = self._vectorize(q_tokens)
        scores = []
        for doc_vec in self.doc_vectors:
            dot = sum(doc_vec.get(term, 0.0) * w for term, w in q_vec.items())
            scores.append(dot)
        return scores

class DenseRetriever:
    _shared_tokenizer = None
    _shared_model = None

    def __init__(self, chunks: List[DocumentChunk], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: Optional[Path] = None):
        self.chunks = chunks
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = DenseRetriever._shared_tokenizer
        self.model = DenseRetriever._shared_model
        self.chunk_embeddings = None
        self.tfidf_fallback = None
        self._init_model_and_embeddings()

    def _init_model_and_embeddings(self):
        cache_file = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"dense_emb_{len(self.chunks)}_{self.model_name.replace('/', '_')}.npy"

        # Check if precomputed embeddings exist
        if cache_file and cache_file.exists():
            print(f"[DenseRetriever] Loading cached embeddings from {cache_file}...")
            try:
                self.chunk_embeddings = np.load(cache_file)
            except Exception as e:
                print(f"[DenseRetriever] Error loading cache {cache_file}: {e}")
                self.chunk_embeddings = None

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            if DenseRetriever._shared_tokenizer is None or DenseRetriever._shared_model is None:
                print(f"[DenseRetriever] Loading {self.model_name} from local cache...")
                try:
                    DenseRetriever._shared_tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                    DenseRetriever._shared_model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
                except Exception as ex:
                    print(f"[DenseRetriever] Local model load failed ({ex}). Falling back to TF-IDF.")
                    raise ex
                DenseRetriever._shared_model.eval()

            self.tokenizer = DenseRetriever._shared_tokenizer
            self.model = DenseRetriever._shared_model
            
            # If embeddings were not cached, compute them now
            if self.chunk_embeddings is None:
                texts = [f"Title: {c.paper_title}. Section: {c.section}.\n{c.text}" for c in self.chunks]
                print(f"[DenseRetriever] Encoding {len(self.chunks)} chunks...")
                
                # Batch encode
                batch_size = 32
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    with torch.no_grad():
                        model_output = self.model(**encoded_input)
                        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
                        # L2 normalize
                        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                        all_embeddings.append(sentence_embeddings.cpu().numpy())

                self.chunk_embeddings = np.vstack(all_embeddings)
                if cache_file:
                    np.save(cache_file, self.chunk_embeddings)
                    print(f"[DenseRetriever] Saved embeddings cache to {cache_file}")

        except Exception as e:
            print(f"[DenseRetriever] Transformers embedding unavailable or offline: {e}. Activating high-speed TF-IDF semantic engine.")
            texts = [f"{c.paper_title} {c.section} {c.text}" for c in self.chunks]
            self.tfidf_fallback = PurePythonTFIDF(texts)

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[RetrievedPassage]:
        if not self.chunks or not query.strip():
            return []

        if self.chunk_embeddings is not None and self.tokenizer is not None and self.model is not None:
            import torch
            encoded_query = self.tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**encoded_query)
                q_emb = mean_pooling(output, encoded_query["attention_mask"])
                q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).cpu().numpy()[0]
            scores = np.dot(self.chunk_embeddings, q_emb)
        elif self.tfidf_fallback is not None:
            scores = np.array(self.tfidf_fallback.score(query))
        else:
            return []

        scored_indices = np.argsort(scores)[::-1]
        results: List[RetrievedPassage] = []
        rank = 1
        for idx in scored_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            chunk = self.chunks[idx]
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
            rank += 1
            if len(results) >= top_k:
                break
        return results
