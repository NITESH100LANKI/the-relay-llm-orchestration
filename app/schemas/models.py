"""
Data schemas and type definitions for The Relay.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class PaperMetadata(BaseModel):
    title: str
    arxiv_id: str
    version: str
    filename: str
    resolved_url: str
    sha256: str
    size_bytes: int
    page_count: Optional[int] = None

class DocumentPage(BaseModel):
    paper_title: str
    arxiv_id: str
    version: str
    page_number: int
    raw_text: str
    clean_text: str
    sections: List[str] = Field(default_factory=list)

class DocumentChunk(BaseModel):
    chunk_id: str
    paper_title: str
    arxiv_id: str
    version: str
    page_number: int
    section: str = "General"
    text: str
    token_count: int

class RetrievalConfig(BaseModel):
    name: str = "default"
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k: int = 3
    rerank_enabled: bool = False
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    min_score_threshold: float = 0.0

class RetrievedPassage(BaseModel):
    chunk_id: str
    paper_title: str
    arxiv_id: str
    version: str
    page_number: int
    section: str
    text: str
    score: float
    rank: int

class Citation(BaseModel):
    paper_title: str
    arxiv_id: str
    version: str
    page_number: int
    section: Optional[str] = None
    supporting_quote: Optional[str] = None

class QuestionUnderstanding(BaseModel):
    raw_question: str
    intent: str
    technical_terms: List[str] = Field(default_factory=list)
    target_papers: List[str] = Field(default_factory=list)
    needs_retrieval: bool = True
    complexity: Literal["simple", "cross_paper", "comparative", "unanswerable"] = "simple"

class PipelineTraceStep(BaseModel):
    step_name: str
    started_at_ms: float
    duration_ms: float
    status: Literal["ok", "skipped", "failed", "retried", "rate_limited", "timeout", "error"] = "ok"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PipelineResult(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None
    retrieved_passages: List[RetrievedPassage] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    first_token_ms: Optional[float] = None
    provider_ttft_ms: Optional[float] = None
    total_ms: float = 0.0
    queue_wait_ms: float = 0.0
    retries: int = 0
    status: Literal["ok", "rate_limited", "timeout", "error"] = "ok"
    traces: List[PipelineTraceStep] = Field(default_factory=list)
    source: Literal["real", "stand-in"] = "stand-in"
    retrieval_setting: str = "A"
    answer_grade: Optional[Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]] = None

class AskRequest(BaseModel):
    question: str
    retrieval_setting: Literal["A", "B", "C"] = "B"
    stream: bool = False
    request_deadline_ms: Optional[float] = 30000.0

class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: List[Citation]
    refused: bool
    retrieval_setting: str
    first_token_ms: Optional[float] = None
    total_ms: float
    tokens_in: int
    tokens_out: int
    retries: int
    status: str
    source: str
