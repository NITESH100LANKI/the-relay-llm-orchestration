"""
Controlled Real Groq API Measurement & Calibration Script for The Relay.
Sends a modest number of controlled requests to measure:
- Real TTFT (Time to first token)
- Real total latency
- Token consumption (prompt & completion)
- Exact rate-limit headers (x-ratelimit-remaining-*, x-ratelimit-limit-*, retry-after)
- Streaming behavior and token arrival pace

Appends results directly to submission/results.csv with source="real".
Redacts all API keys and sensitive tokens.
"""

import os
import sys
import time
import json
import csv
import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

from app.schemas.models import AskRequest, PipelineResult
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.groq_provider import GroqProvider
from app.evaluation.benchmark_questions import BENCHMARK_QUESTIONS

RESULTS_CSV_PATH = BASE_DIR / "submission" / "results.csv"
LOGS_DIR = BASE_DIR / "experiments" / "real_service"

def append_to_csv(result: PipelineResult, scenario: str):
    fieldnames = [
        "request_id",
        "scenario",
        "source",
        "start_time",
        "first_token_ms",
        "total_ms",
        "status",
        "retries",
        "tokens_in",
        "tokens_out",
        "retrieval_setting",
        "answer_grade"
    ]
    row = {
        "request_id": result.request_id,
        "scenario": scenario,
        "source": "real",
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "first_token_ms": round(result.first_token_ms, 1),
        "total_ms": round(result.total_ms, 1),
        "status": result.status,
        "retries": result.retries,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "retrieval_setting": result.retrieval_setting,
        "answer_grade": result.answer_grade or "wrong"
    }
    with open(RESULTS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
    print(f"Recorded real request {row['request_id'][:8]}... Status: {row['status']} | TTFT: {row['first_token_ms']}ms | Total: {row['total_ms']}ms")

async def test_single_direct_call(provider: GroqProvider):
    print("\n--- [Experiment 1: Single Direct Groq Call & Header Capture] ---")
    prompt = "In Attention Is All You Need, what is Scaled Dot-Product Attention? Answer in 2 concise sentences."
    t0 = time.perf_counter()
    resp = await provider.generate(prompt=prompt)
    t1 = time.perf_counter()
    
    print(f"Status: {resp.status}")
    print(f"Duration: {(t1-t0)*1000:.1f} ms")
    print(f"Tokens in: {resp.tokens_in}, Tokens out: {resp.tokens_out}")
    print(f"Response snippet: {resp.content[:120]}...")
    print("\nCaptured Rate-Limit Headers from Groq:")
    for k, v in resp.rate_limit_headers.items():
        print(f"  {k}: {v}")

    return resp

async def test_real_streaming_call(provider: GroqProvider):
    print("\n--- [Experiment 2: Streaming TTFT & Token Pace Measurement] ---")
    prompt = "Explain the Mimi audio codec in Moshi in 3 sentences."
    t0 = time.perf_counter()
    chunks = []
    first_ms = None
    
    async for chunk in provider.generate_stream(prompt=prompt):
        if chunk["is_first"]:
            first_ms = (time.perf_counter() - t0) * 1000.0
            print(f"-> First Token Received at: {first_ms:.1f} ms")
        if chunk["delta"]:
            chunks.append(chunk["delta"])
            sys.stdout.write(chunk["delta"])
            sys.stdout.flush()
        if chunk["done"]:
            final_resp = chunk["response"]

    tot_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\n\nTotal Streaming Latency: {tot_ms:.1f} ms")
    print(f"Tokens Out: {final_resp.tokens_out if final_resp else len(chunks)}")
    return first_ms, tot_ms, final_resp

async def test_real_pipeline_queries(orchestrator: PipelineOrchestrator, count: int = 5):
    print(f"\n--- [Experiment 3: Controlled End-to-End Real Pipeline Queries (n={count})] ---")
    for i in range(count):
        q_item = BENCHMARK_QUESTIONS[i]
        print(f"\nSubmitting Query {i+1}/{count}: '{q_item['question'][:60]}...'")
        req = AskRequest(question=q_item["question"], retrieval_setting="B", stream=False)
        res = await orchestrator.execute(req)
        append_to_csv(res, "real_service_controlled")
        print(f" -> Grade: {res.answer_grade} | Citations: {len(res.citations)}")
        for c in res.citations:
            print(f"    * {c.paper_title}, Page {c.page_number}")
        # Modest pacing between requests to stay well within 30 RPM
        await asyncio.sleep(2.0)

async def main():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or not api_key.startswith("gsk_"):
        print("Error: Valid GROQ_API_KEY not found in environment.")
        return

    provider = GroqProvider(api_key=api_key, model=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"))
    orchestrator = PipelineOrchestrator(provider=provider)

    # 1. Direct call & header inspection
    resp = await test_single_direct_call(provider)

    # 2. Streaming TTFT pace
    await test_real_streaming_call(provider)

    # 3. Controlled real pipeline test (5 sample questions across papers)
    await test_real_pipeline_queries(orchestrator, count=5)

    print("\n[Real Service Experiments Completed Successfully!]")

if __name__ == "__main__":
    asyncio.run(main())
