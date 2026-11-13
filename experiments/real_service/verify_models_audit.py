"""
Independent Technical Audit Script for Real Groq Integration.
Verifies both active model IDs:
1. qwen/qwen3.8-27b
2. openai/gpt-oss-120b

Captures real HTTP headers, TTFT, total latency, token usage,
and distinguishes RPD vs TPM vs RPM correctly.
"""

import os
import sys
import time
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.environ.get("GROQ_API_KEY", "")
MODELS_TO_VERIFY = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]

def mask_key(k: str) -> str:
    if len(k) > 10:
        return k[:6] + "..." + k[-4:]
    return "***"

def extract_ratelimit_headers(headers: httpx.Headers) -> dict:
    return {k.lower(): v for k, v in headers.items() if k.lower().startswith("x-ratelimit-") or k.lower() == "retry-after"}

def test_model(model_id: str):
    print(f"\n=======================================================")
    print(f" AUDITING GROQ MODEL: {model_id}")
    print(f" Key: {mask_key(API_KEY)}")
    print(f"=======================================================")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 1. Non-streaming call
    payload_sync = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a concise research assistant."},
            {"role": "user", "content": "What is the formula for scaled dot-product attention in Attention Is All You Need?"}
        ],
        "max_tokens": 150,
        "temperature": 0.1,
        "stream": False
    }
    
    t0 = time.perf_counter()
    with httpx.Client(timeout=30.0) as client:
        resp = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload_sync)
    total_ms = (time.perf_counter() - t0) * 1000.0
    
    rl_headers = extract_ratelimit_headers(resp.headers)
    print(f"HTTP Status: {resp.status_code}")
    print(f"Total Non-Streaming Latency: {total_ms:.1f} ms")
    print(f"Rate-Limit Headers:")
    for hk, hv in rl_headers.items():
        print(f"  {hk}: {hv}")
        
    if resp.status_code == 200:
        data = resp.json()
        usage = data.get("usage", {})
        print(f"Prompt Tokens: {usage.get('prompt_tokens')}, Completion Tokens: {usage.get('completion_tokens')}")
        print(f"Snippet: {data['choices'][0]['message']['content'][:120]}...")
    else:
        print(f"Error Body: {resp.text}")

    # 2. Streaming call to measure true TTFT
    payload_stream = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a concise research assistant."},
            {"role": "user", "content": "Name the audio codec in Moshi and its bitrate."}
        ],
        "max_tokens": 150,
        "temperature": 0.1,
        "stream": True
    }
    
    t_stream_start = time.perf_counter()
    ttft_ms = None
    stream_chunks = []
    
    with httpx.Client(timeout=30.0) as client:
        with client.stream("POST", "https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload_stream) as stream_resp:
            for line in stream_resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                d = line[6:].strip()
                if d == "[DONE]":
                    break
                try:
                    c = json.loads(d)
                    delta = c["choices"][0]["delta"].get("content", "")
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_stream_start) * 1000.0
                        stream_chunks.append(delta)
                except Exception:
                    pass
                    
    total_stream_ms = (time.perf_counter() - t_stream_start) * 1000.0
    print(f"\nStreaming Results:")
    print(f"  TTFT: {ttft_ms:.1f} ms" if ttft_ms else "  TTFT: None")
    print(f"  Total Stream Time: {total_stream_ms:.1f} ms")
    print(f"  Content: {''.join(stream_chunks)[:120]}...")
    
    return {
        "model_id": model_id,
        "status_code": resp.status_code,
        "non_stream_latency_ms": round(total_ms, 1),
        "streaming_ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "streaming_total_ms": round(total_stream_ms, 1),
        "headers": rl_headers,
        "content_sync": data['choices'][0]['message']['content'] if resp.status_code == 200 else resp.text,
        "content_stream": "".join(stream_chunks)
    }

def main():
    results = {}
    for m in MODELS_TO_VERIFY:
        results[m] = test_model(m)
        time.sleep(2.0)  # Gentle pause between model calls
        
    audit_file = BASE_DIR / "experiments" / "real_service" / "models_audit_log.json"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote complete audit results to {audit_file}")

if __name__ == "__main__":
    main()

