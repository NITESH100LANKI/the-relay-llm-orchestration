"""
Configurable Document Chunker for The Relay.
Splits extracted document pages into overlapping text chunks while strictly
preserving metadata:
- Paper title
- arXiv ID
- Exact version
- Page number
- Section heading
- Deterministic chunk ID
"""

import re
from typing import List
from app.schemas.models import DocumentPage, DocumentChunk, RetrievalConfig

def estimate_tokens(text: str) -> int:
    """
    Fast, standard token count approximation:
    Counts word tokens plus punctuation marks.
    Approximately 1.3 tokens per whitespace-separated word for technical English.
    """
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return len(words)

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences, being careful not to split on common abbreviations (e.g. Fig., Eq., et al., e.g., i.e.).
    """
    # Replace common abbreviations periods temporarily
    protected = text
    abbrevs = [
        ("Fig.", "Fig@"),
        ("Eq.", "Eq@"),
        ("al.", "al@"),
        ("e.g.", "eg@"),
        ("i.e.", "ie@"),
        ("Sec.", "Sec@"),
        ("Ref.", "Ref@"),
        ("vs.", "vs@"),
        ("Tab.", "Tab@"),
        ("Vol.", "Vol@"),
        ("No.", "No@")
    ]
    for orig, rep in abbrevs:
        protected = protected.replace(orig, rep)
        
    raw_sentences = re.split(r"(?<=[.!?])\s+", protected)
    sentences = []
    for s in raw_sentences:
        for orig, rep in abbrevs:
            s = s.replace(rep, orig)
        s_clean = s.strip()
        if s_clean:
            sentences.append(s_clean)
            
    return sentences

def chunk_pages(
    pages: List[DocumentPage],
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> List[DocumentChunk]:
    """
    Chunks document pages into overlapping chunks of approximately chunk_size tokens,
    with chunk_overlap tokens overlap, while retaining page numbers and section metadata.
    """
    chunks: List[DocumentChunk] = []
    
    # We group by paper
    papers_map = {}
    for p in pages:
        key = (p.paper_title, p.arxiv_id, p.version)
        if key not in papers_map:
            papers_map[key] = []
        papers_map[key].append(p)
        
    for (title, arxiv_id, version), paper_pages in papers_map.items():
        chunk_idx = 0
        for page in paper_pages:
            text = page.clean_text
            if not text.strip():
                continue
                
            sentences = split_into_sentences(text)
            if not sentences:
                continue
                
            current_sentences: List[str] = []
            current_tokens = 0
            
            section_title = page.sections[0] if page.sections else "General"
            
            for sent in sentences:
                sent_tokens = estimate_tokens(sent)
                
                # If a single sentence is larger than chunk_size, split by words
                if sent_tokens > chunk_size:
                    words = sent.split()
                    step = chunk_size - chunk_overlap
                    if step <= 0:
                        step = max(1, chunk_size // 2)
                    for i in range(0, len(words), step):
                        sub_words = words[i:i + chunk_size]
                        sub_text = " ".join(sub_words)
                        chunk_idx += 1
                        chunk_id = f"{arxiv_id}v{version}_p{page.page_number}_c{chunk_idx:04d}"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                paper_title=title,
                                arxiv_id=arxiv_id,
                                version=version,
                                page_number=page.page_number,
                                section=section_title,
                                text=sub_text,
                                token_count=len(sub_words)
                            )
                        )
                    continue

                if current_tokens + sent_tokens > chunk_size and current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunk_idx += 1
                    chunk_id = f"{arxiv_id}v{version}_p{page.page_number}_c{chunk_idx:04d}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            paper_title=title,
                            arxiv_id=arxiv_id,
                            version=version,
                            page_number=page.page_number,
                            section=section_title,
                            text=chunk_text,
                            token_count=current_tokens
                        )
                    )
                    
                    # Compute overlap by keeping tail sentences
                    overlap_sentences = []
                    overlap_tokens = 0
                    for s in reversed(current_sentences):
                        s_tok = estimate_tokens(s)
                        if overlap_tokens + s_tok <= chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tok
                        else:
                            break
                    current_sentences = overlap_sentences
                    current_tokens = overlap_tokens

                current_sentences.append(sent)
                current_tokens += sent_tokens

            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunk_idx += 1
                chunk_id = f"{arxiv_id}v{version}_p{page.page_number}_c{chunk_idx:04d}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        paper_title=title,
                        arxiv_id=arxiv_id,
                        version=version,
                        page_number=page.page_number,
                        section=section_title,
                        text=chunk_text,
                        token_count=current_tokens
                    )
                )

    return chunks

