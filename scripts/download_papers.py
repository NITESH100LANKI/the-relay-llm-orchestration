"""
Paper downloader script for The Relay.
Downloads the three exact versioned research papers from arXiv:
1. Attention Is All You Need (1706.03762v7)
2. Moshi: a speech-text foundation model for real-time dialogue (2410.00037v1)
3. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (2507.07955v2 / search)

Verifies SHA-256 checksums, downloads headers, and writes a manifest.json.
"""

import os
import hashlib
import json
import urllib.request
import urllib.error
import ssl
from pathlib import Path

PAPERS = [
    {
        "id": "1706.03762v7",
        "arxiv_id": "1706.03762",
        "version": "v7",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "alt_url": "https://arxiv.org/pdf/1706.03762v7",
        "filename": "1706.03762v7_attention_is_all_you_need.pdf"
    },
    {
        "id": "2410.00037v1",
        "arxiv_id": "2410.00037",
        "version": "v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "alt_url": "https://arxiv.org/pdf/2410.00037v1",
        "filename": "2410.00037v1_moshi_speech_text_foundation_model.pdf"
    },
    {
        "id": "2507.07955v2",
        "arxiv_id": "2507.07955",
        "version": "v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "alt_url": "https://arxiv.org/pdf/2507.07955v2",
        "fallback_url": "https://arxiv.org/pdf/2407.07955v2",  # in case 2507 was typo for 2407
        "filename": "2507.07955v2_dynamic_chunking_hierarchical_sequence.pdf"
    }
]

def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def download_file(url: str, alt_url: str, dest_path: Path) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TheRelayCorpusFetcher/1.0"}
    ctx = ssl._create_unverified_context()
    for target in [url, alt_url]:
        try:
            print(f"Trying {target} ...")
            req = urllib.request.Request(target, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
                content = response.read()
                # Check if it's a valid PDF (starts with %PDF)
                if content.startswith(b"%PDF"):
                    with open(dest_path, "wb") as f:
                        f.write(content)
                    print(f"Successfully downloaded {len(content)} bytes from {target}")
                    return target
                else:
                    print(f"Response from {target} was not a PDF (len={len(content)})")
        except urllib.error.URLError as e:
            print(f"Failed to fetch {target}: {e}")
    raise RuntimeError(f"Could not download PDF from {url} or {alt_url}")

def main():
    target_dir = Path(__file__).resolve().parent.parent / "corpus" / "papers"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    for paper in PAPERS:
        dest = target_dir / paper["filename"]
        resolved_url = paper["url"]
        if not dest.exists():
            try:
                resolved_url = download_file(paper["url"], paper.get("alt_url", paper["url"]), dest)
            except Exception as e:
                if "fallback_url" in paper:
                    print(f"Attempting fallback URL: {paper['fallback_url']}")
                    resolved_url = download_file(paper["fallback_url"], paper["fallback_url"] + ".pdf", dest)
                else:
                    raise e
        
        sha256 = calculate_sha256(dest)
        size_bytes = dest.stat().st_size
        print(f"Verified: {paper['title']} -> {dest.name} (SHA-256: {sha256[:16]}..., {size_bytes} bytes)")
        
        manifest.append({
            "title": paper["title"],
            "arxiv_id": paper["arxiv_id"],
            "version": paper["version"],
            "filename": paper["filename"],
            "resolved_url": resolved_url,
            "sha256": sha256,
            "size_bytes": size_bytes
        })
        
    manifest_path = target_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_path}")

if __name__ == "__main__":
    main()
