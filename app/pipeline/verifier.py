"""
Stage 5: Answer & Citation Verification Module for The Relay.
Verifies:
1. Citations point to real, retrieved passages.
2. Page numbers and arXiv IDs match the source library.
3. Claims are grounded in retrieved evidence.
4. Unsupported questions are appropriately refused.
"""

import re
from typing import List, Tuple, Optional
from app.schemas.models import RetrievedPassage, Citation

CITATION_REGEX = re.compile(
    r"\[(?:Citation:\s*)?([^,\]]+),\s*(?:arXiv\s*)?([0-9\.]+v?[0-9]*),\s*(?:Page\s*(\d+))(?:\s*,\s*Section\s*([^\]]+))?\]",
    re.IGNORECASE
)

REFUSAL_KEYWORDS = [
    "insufficient evidence",
    "not enough information",
    "does not contain",
    "cannot be answered",
    "not mentioned in the provided papers",
    "not discussed",
    "does not provide"
]

def parse_citations(text: str) -> List[Citation]:
    citations = []
    matches = CITATION_REGEX.findall(text)
    for m in matches:
        title = m[0].strip()
        arxiv_full = m[1].strip()
        page = int(m[2].strip()) if m[2] else 1
        section = m[3].strip() if len(m) > 3 and m[3] else None
        
        # Split arxiv id and version
        if "v" in arxiv_full:
            parts = arxiv_full.split("v")
            arxiv_id = parts[0]
            version = "v" + parts[1]
        else:
            arxiv_id = arxiv_full
            version = "v1"

        citations.append(
            Citation(
                paper_title=title,
                arxiv_id=arxiv_id,
                version=version,
                page_number=page,
                section=section
            )
        )
    return citations

def is_refusal(text: str) -> bool:
    t_lower = text.lower()
    return any(kw in t_lower for kw in REFUSAL_KEYWORDS)

def verify_answer(
    answer: str,
    retrieved_passages: List[RetrievedPassage],
    is_expected_unanswerable: bool = False
) -> Tuple[bool, List[Citation], Optional[str], str]:
    """
    Verifies answer correctness and citation validity.
    Returns:
        (is_valid, valid_citations, refusal_reason, answer_grade)
    """
    # 1. Check refusal behavior
    refused = is_refusal(answer)
    if is_expected_unanswerable or not retrieved_passages:
        if refused:
            return True, [], "Correctly refused due to missing or insufficient evidence in the library.", "refused_correctly"
        else:
            # Hallucinated an answer for an unanswerable question
            return False, [], "Hallucination: answered an unsupported question without grounding.", "wrong"

    if refused:
        return True, [], "Assistant determined retrieved evidence was insufficient.", "refused_correctly"

    # 2. Extract citations
    citations = parse_citations(answer)
    if not citations:
        # Substantive answer without citation is invalid
        return False, [], "Answer lacks mandatory citations to the source papers.", "wrong"

    # 3. Verify citations against retrieved passages
    valid_citations = []
    retrieved_pages = {(p.arxiv_id, p.page_number) for p in retrieved_passages}
    retrieved_titles = {p.paper_title.lower() for p in retrieved_passages}

    for c in citations:
        # Check title similarity
        title_matches = any(c.paper_title.lower() in t or t in c.paper_title.lower() for t in retrieved_titles)
        page_matches = (c.arxiv_id, c.page_number) in retrieved_pages
        
        # Give leeway of +-1 page for multi-page sections if title matches
        if not page_matches and title_matches:
            nearby_pages = {(c.arxiv_id, c.page_number - 1), (c.arxiv_id, c.page_number + 1)}
            if nearby_pages.intersection(retrieved_pages):
                page_matches = True

        if title_matches and page_matches:
            valid_citations.append(c)

    if not valid_citations:
        return False, citations, "Citations do not match any retrieved passages.", "wrong"

    return True, valid_citations, None, "correct"

