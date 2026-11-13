"""
PDF Text and Structure Extractor for The Relay.
Uses PyMuPDF (fitz) to extract text while preserving:
- Page numbers
- Section headings
- Mathematical expressions and symbols
- Technical terminology
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF

from app.schemas.models import DocumentPage, PaperMetadata

SECTION_REGEX = re.compile(
    r"^(?:(?:\d+\.?)+\s+[A-Z][A-Za-z0-9\s,\-:]{2,60}|Abstract|Introduction|Conclusion|References|Acknowledgement[s]?|Appendix.*)$",
    re.MULTILINE
)

def clean_extracted_text(text: str) -> str:
    """
    Cleans text while preserving math, hyphenated technical terms, and punctuation.
    Avoids destroying formulas or symbols.
    """
    # Replace weird PDF whitespace artifacts with standard space
    cleaned = text.replace("\xa0", " ").replace("\u200b", "")
    
    # Normalize multiple blank lines to a double newline (paragraph boundary)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    # Normalize excessive inline spaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    
    # Fix broken hyphenation at line breaks (e.g., "atten-\ntion" -> "attention")
    # Only join if followed immediately by lowercase letter
    cleaned = re.sub(r"([a-zA-Z]{2,})-\n([a-z]{2,})", r"\1\2", cleaned)
    
    # Join single line breaks that don't look like section boundaries or lists
    lines = cleaned.split("\n")
    processed_lines = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            processed_lines.append("")
        else:
            processed_lines.append(line_str)
            
    return "\n".join(processed_lines).strip()

def extract_paper(pdf_path: Path, metadata: Dict[str, Any]) -> List[DocumentPage]:
    """
    Extracts all pages from a PDF, identifying section headers and page metadata.
    """
    doc = fitz.open(pdf_path)
    pages: List[DocumentPage] = []
    
    current_section = "Abstract / Header"
    
    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]
        
        # Extract text blocks
        # blocks format: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        
        page_text_pieces = []
        detected_sections_on_page = []
        
        for b in blocks:
            # block_type 0 is text, 1 is image
            if b[6] == 0:
                block_text = b[4].strip()
                if not block_text:
                    continue
                
                # Check if block looks like a section heading
                first_line = block_text.split("\n")[0].strip()
                if len(first_line) < 70 and SECTION_REGEX.match(first_line):
                    current_section = first_line
                    detected_sections_on_page.append(current_section)
                
                page_text_pieces.append(block_text)
                
        raw_page_text = "\n\n".join(page_text_pieces)
        cleaned_page_text = clean_extracted_text(raw_page_text)
        
        pages.append(
            DocumentPage(
                paper_title=metadata["title"],
                arxiv_id=metadata["arxiv_id"],
                version=metadata["version"],
                page_number=page_num,
                raw_text=raw_page_text,
                clean_text=cleaned_page_text,
                sections=detected_sections_on_page if detected_sections_on_page else [current_section]
            )
        )
        
    doc.close()
    return pages

def extract_corpus(manifest_path: Path, output_dir: Path) -> List[DocumentPage]:
    """
    Extracts all papers specified in the manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    all_pages: List[DocumentPage] = []
    papers_dir = manifest_path.parent
    
    for item in manifest:
        pdf_path = papers_dir / item["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")
            
        print(f"Extracting {item['title']} ({item['arxiv_id']}{item['version']}) from {pdf_path.name}...")
        paper_pages = extract_paper(pdf_path, item)
        item["page_count"] = len(paper_pages)
        print(f" -> Extracted {len(paper_pages)} pages.")
        all_pages.extend(paper_pages)
        
    # Save processed pages
    output_path = output_dir / "extracted_pages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in all_pages], f, indent=2)
        
    # Update manifest with page counts
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Successfully extracted {len(all_pages)} total pages to {output_path}")
    return all_pages

if __name__ == "__main__":
    from app.core.config import settings
    extract_corpus(settings.corpus_dir / "manifest.json", settings.processed_dir)

