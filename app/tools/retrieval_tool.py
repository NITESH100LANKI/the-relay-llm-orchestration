"""
Explicit Retrieval Tool Invocation for The Relay AI Pipeline.
Defines a callable tool interface with schemas, parameter validation,
and execution telemetry.
"""

import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from app.schemas.models import RetrievedPassage, RetrievalConfig
from app.retrieval.engine import retrieve_paper_passages

class RetrievalToolInput(BaseModel):
    query: str = Field(description="The technical search query to retrieve passages for.")
    retrieval_setting: str = Field(default="B", description="The retrieval preset: 'A', 'B', or 'C'.")
    strategy: str = Field(default="primary", description="'primary' or 'secondary_fallback'")

class RetrievalToolOutput(BaseModel):
    tool_name: str = "retrieve_paper_passages"
    execution_time_ms: float
    passages_count: int
    passages: List[RetrievedPassage]
    strategy_used: str

class RetrievalTool:
    name: str = "retrieve_paper_passages"
    description: str = (
        "Searches the indexed research paper library (Attention Is All You Need, Moshi, Dynamic Chunking) "
        "and returns ranked passages with verified paper titles, arXiv IDs, versions, page numbers, and sections."
    )

    async def execute(self, tool_input: RetrievalToolInput) -> RetrievalToolOutput:
        t0 = time.perf_counter()
        query = tool_input.query
        
        # If secondary strategy is specified, refine query or use reranking preset C
        setting = tool_input.retrieval_setting
        if tool_input.strategy == "secondary_fallback":
            setting = "C"  # Escalate to hybrid RRF reranker
            
        passages = retrieve_paper_passages(query, setting)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        
        return RetrievalToolOutput(
            tool_name=self.name,
            execution_time_ms=duration_ms,
            passages_count=len(passages),
            passages=passages,
            strategy_used=f"{setting}_{tool_input.strategy}"
        )

