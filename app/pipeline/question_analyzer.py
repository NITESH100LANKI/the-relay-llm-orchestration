"""
Stage 1: Question Understanding Module for The Relay.
Analyzes user questions to:
- Identify query intent
- Extract technical terminology
- Detect targeted papers
- Flag questions clearly outside the research corpus scope
"""

import re
from typing import List
from app.schemas.models import QuestionUnderstanding

PAPER_TOPICS = {
    "1706.03762": {
        "title": "Attention Is All You Need",
        "keywords": ["transformer", "attention", "multi-head", "dot-product", "bleu", "vaswani", "encoder", "decoder", "self-attention", "sinusoid", "positional"]
    },
    "2410.00037": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "keywords": ["moshi", "mimi", "speech", "dialogue", "duplex", "audio", "codec", "kyutai", "latency", "streaming", "rvq", "acoustic"]
    },
    "2507.07955": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "keywords": ["dynamic chunking", "hierarchical", "sequence modeling", "chunk", "entropy", "boundary", "compression", "multiscale", "downsampling"]
    }
}

OUT_OF_SCOPE_PATTERNS = [
    r"\bcost\b.*(?:dollar|usd|\$|price|subscription|bill|budget|hardware\s+purchase)",
    r"\bstock\s+price\b",
    r"\bweather\b",
    r"\bwho\s+won\b",
    r"\brecipe\b",
    r"\bpython\s+code\s+for\s+django\b",
    r"\bquantum\s+computing\b",
    r"\bhow\s+to\s+cook\b"
]

def analyze_question(question: str) -> QuestionUnderstanding:
    q_lower = question.lower()
    
    # Check out-of-scope triggers
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, q_lower):
            return QuestionUnderstanding(
                raw_question=question,
                intent="out_of_scope_query",
                technical_terms=[],
                target_papers=[],
                needs_retrieval=False,
                complexity="unanswerable"
            )

    # Extract technical terms
    detected_terms = []
    matched_papers = []
    
    for arxiv_id, data in PAPER_TOPICS.items():
        found = False
        for kw in data["keywords"]:
            if kw in q_lower:
                detected_terms.append(kw)
                found = True
        if found:
            matched_papers.append(data["title"])

    complexity = "simple"
    if len(matched_papers) > 1 or "compare" in q_lower or "difference" in q_lower:
        complexity = "comparative"
    elif not detected_terms and len(question.split()) > 3:
        complexity = "unanswerable"

    return QuestionUnderstanding(
        raw_question=question,
        intent="paper_qa",
        technical_terms=list(set(detected_terms)),
        target_papers=matched_papers,
        needs_retrieval=True,
        complexity=complexity
    )

