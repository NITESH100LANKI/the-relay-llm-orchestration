"""
Real Groq Retrieval Quality Evaluation Runner.
Evaluates the 15 self-authored benchmark questions across Configurations A, B, and C
using the real Groq LLM (qwen/qwen3.8-27b) with adaptive token-bucket pacing.
Records granular per-question metadata and produces verified Table 3 metrics.
"""

import os
import sys
import time
import json
import csv
import re
import math
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from app.schemas.models import AskRequest, PipelineResult
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.groq_provider import GroqProvider
from app.evaluation.benchmark_questions import BENCHMARK_QUESTIONS
from app.core.config import settings

OUTPUT_JSON = settings.base_dir / "submission" / "retrieval_benchmark_real_groq.json"
TABLE_3_CSV = settings.base_dir / "submission" / "tables" / "table_3.csv"

def parse_reset_time_to_seconds(reset_str: str) -> float:
    """Parses Groq reset header like '1.425s', '480ms', '2m52.8s' into float seconds."""
    if not reset_str:
        return 2.0
    reset_str = reset_str.strip()
    try:
        if reset_str.endswith("ms"):
            return max(0.5, float(reset_str[:-2]) / 1000.0)
        elif reset_str.endswith("s") and "m" not in reset_str:
            return max(0.5, float(reset_str[:-1]))
        elif "m" in reset_str:
            parts = reset_str.split("m")
            m = float(parts[0])
            s = float(parts[1].replace("s", "")) if len(parts) > 1 and parts[1] else 0.0
            return max(1.0, m * 60.0 + s)
    except Exception:
        pass
    return 2.0

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    d0 = sorted_v[int(f)] * (c - k)
    d1 = sorted_v[int(c)] * (k - f)
    return d0 + d1

async def run_benchmark():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    TABLE_3_CSV.parent.mkdir(parents=True, exist_ok=True)

    provider = GroqProvider()
    orchestrator = PipelineOrchestrator(provider=provider)

    print(f"=================================================================")
    print(f"STARTING REAL GROQ RETRIEVAL EVALUATION")
    print(f"Model: {provider.model}")
    print(f"Configurations: A (300tok/3pcs), B (500tok/5pcs), C (500tok/5pcs RRF)")
    print(f"Questions: 15 self-authored benchmark questions")
    print(f"Output JSON: {OUTPUT_JSON}")
    print(f"=================================================================\n")

    # Warm up dense index once
    print("[Warmup] Initializing and warming up dense retrieval index...")
    t_warm = time.time()
    from app.retrieval.engine import retrieve_paper_passages
    _ = retrieve_paper_passages("Attention warmup test", "B")
    print(f"[Warmup] Index ready in {time.time() - t_warm:.2f}s.\n")

    all_records: List[Dict[str, Any]] = []

    # Check if existing partial records exist to allow resuming
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                all_records = json.load(f)
            print(f"[Resume] Loaded {len(all_records)} existing records from {OUTPUT_JSON}")
        except Exception:
            all_records = []

    existing_keys = {(r["retrieval_setting"], r["question_id"]) for r in all_records}

    for setting in ["A", "B", "C"]:
        print(f"\n=======================================================")
        print(f">>> EVALUATING RETRIEVAL SETTING {setting} (Real Groq: {provider.model})")
        print(f"=======================================================")

        for idx, q_item in enumerate(BENCHMARK_QUESTIONS):
            q_id = q_item["id"]
            if (setting, q_id) in existing_keys:
                print(f"[{setting} | {q_id}] Already evaluated. Skipping.")
                continue

            q_text = q_item["question"]
            expected_grade = q_item["expected_grade"]
            category = q_item["category"]
            target_paper = q_item["target_paper"]

            print(f"\n[{setting} | {q_id}] ({category}) Q: {q_text[:75]}...")
            req = AskRequest(question=q_text, retrieval_setting=setting)
            
            t_req_start = time.perf_counter()
            res = await orchestrator.execute(req)
            elapsed_sec = time.perf_counter() - t_req_start

            record = {
                "question_id": q_id,
                "category": category,
                "question": q_text,
                "expected_grade": expected_grade,
                "target_paper": target_paper,
                "retrieval_setting": setting,
                "model_id": provider.model,
                "status": res.status,
                "answer_grade": res.answer_grade,
                "refused": res.refused,
                "refusal_reason": res.refusal_reason,
                "answer": res.answer,
                "citations": [
                    {
                        "paper_title": c.paper_title,
                        "arxiv_id": c.arxiv_id,
                        "version": c.version,
                        "page_number": c.page_number,
                        "section": c.section
                    }
                    for c in res.citations
                ],
                "retrieved_passages": [
                    {
                        "chunk_id": p.chunk_id,
                        "paper_title": p.paper_title,
                        "page_number": p.page_number,
                        "section": p.section
                    }
                    for p in res.retrieved_passages
                ],
                "tokens_in": res.tokens_in,
                "tokens_out": res.tokens_out,
                "first_token_ms": res.first_token_ms,
                "provider_ttft_ms": res.provider_ttft_ms,
                "total_ms": res.total_ms,
                "retries": res.retries,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rate_limit_headers": provider.last_rate_limit_headers
            }

            all_records.append(record)
            existing_keys.add((setting, q_id))

            print(f"   Status: {res.status} | Grade: {res.answer_grade} | Refused: {res.refused}")
            print(f"   Tokens: In={res.tokens_in}, Out={res.tokens_out} | TTFT: {res.first_token_ms}ms | Total: {res.total_ms:.1f}ms")
            if res.citations:
                for c in res.citations[:2]:
                    print(f"   Citation: {c.paper_title} (p.{c.page_number}, sec={c.section})")
            elif res.refused:
                print(f"   Refusal: {res.answer[:90]}...")

            # Save progress to JSON immediately
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2)

            # Adaptive Pacing
            rl_headers = provider.last_rate_limit_headers
            rem_tokens_str = rl_headers.get("x-ratelimit-remaining-tokens", "8000")
            reset_tokens_str = rl_headers.get("x-ratelimit-reset-tokens", "2.0s")
            rem_requests_str = rl_headers.get("x-ratelimit-remaining-requests", "1000")

            try:
                rem_tokens = int(rem_tokens_str)
            except Exception:
                rem_tokens = 8000

            reset_sec = parse_reset_time_to_seconds(reset_tokens_str)
            print(f"   [Groq Rate Limits] Rem Tokens: {rem_tokens}/8000 | Rem Reqs: {rem_requests_str} | Token Reset: {reset_tokens_str}")

            # If remaining tokens are low, sleep for token replenishment window
            if rem_tokens < 3500:
                sleep_time = reset_sec + 2.5
                print(f"   [Pacing] Low remaining tokens ({rem_tokens}). Sleeping {sleep_time:.1f}s for replenishment...")
                await asyncio.sleep(sleep_time)
            else:
                # Setting A uses fewer tokens (~1100), B/C use ~2200 tokens
                base_sleep = 5.0 if setting == "A" else 8.5
                await asyncio.sleep(base_sleep)

    # =========================================================================
    # Compute Final Reconciled Table 3
    # =========================================================================
    print("\n=======================================================")
    print("COMPUTING FINAL RECONCILED TABLE 3 METRICS")
    print("=======================================================")

    config_meta = [
        ("A", "300 tokens, 3 pieces, re-rank no"),
        ("B", "500 tokens, 5 pieces, re-rank no"),
        ("C", "500 tokens, 5 pieces, re-rank yes")
    ]

    table_3_rows = []

    for s_code, s_label in config_meta:
        s_items = [r for r in all_records if r["retrieval_setting"] == s_code]
        n_attempted = len(s_items)
        if n_attempted == 0:
            continue

        n_ok = len([r for r in s_items if r["status"] == "ok"])
        n_rate_limited = len([r for r in s_items if r["status"] == "rate_limited"])
        
        # Grading counts over actual evaluated questions
        n_correct = len([r for r in s_items if r["answer_grade"] == "correct"])
        n_wrong = len([r for r in s_items if r["answer_grade"] == "wrong"])
        n_refused_correctly = len([r for r in s_items if r["answer_grade"] == "refused_correctly"])
        n_refused_wrongly = len([r for r in s_items if r["answer_grade"] == "refused_wrongly"])

        pct_correct = (n_correct / n_attempted) * 100.0
        pct_wrong = (n_wrong / n_attempted) * 100.0
        pct_refused_correctly = (n_refused_correctly / n_attempted) * 100.0

        # Mean tokens per SUCCESSFUL answer (do NOT dilute with zero-token failures!)
        successful_records = [r for r in s_items if r["status"] == "ok" and (r["tokens_in"] + r["tokens_out"] > 0)]
        n_success = len(successful_records)

        if n_success > 0:
            avg_tokens_in = sum(r["tokens_in"] for r in successful_records) / n_success
            avg_tokens_out = sum(r["tokens_out"] for r in successful_records) / n_success
            avg_total_tokens = avg_tokens_in + avg_tokens_out
        else:
            avg_tokens_in = 0.0
            avg_tokens_out = 0.0
            avg_total_tokens = 0.0

        # Valid TTFT strictly over requests that emitted tokens
        valid_ttfts = [r["first_token_ms"] for r in successful_records if r["first_token_ms"] is not None]
        ttft_p50 = percentile(valid_ttfts, 50) if valid_ttfts else 0.0
        ttft_p95 = percentile(valid_ttfts, 95) if valid_ttfts else 0.0

        # Total latency over successful requests
        totals = [r["total_ms"] for r in s_items]
        tot_p50 = percentile(totals, 50) if totals else 0.0
        tot_p95 = percentile(totals, 95) if totals else 0.0

        row = {
            "Setting (piece size, pieces fetched, re-rank yes or no)": s_label,
            "Questions asked": n_attempted,
            "Correct and cited %": f"{pct_correct:.1f}%",
            "Wrong %": f"{pct_wrong:.1f}%",
            "Correctly refused %": f"{pct_refused_correctly:.1f}%",
            "Tokens per answer": round(avg_total_tokens),
            "First token p50 / p95 (ms)": f"{ttft_p50:.1f} / {ttft_p95:.1f}",
            "Total time p50 / p95 (ms)": f"{tot_p50:.1f} / {tot_p95:.1f}",
            "Questions attempted": n_attempted,
            "Successfully answered": n_ok,
            "Correctly refused": n_refused_correctly,
            "Rate-limited": n_rate_limited,
            "Mean prompt tokens": round(avg_tokens_in),
            "Mean completion tokens": round(avg_tokens_out)
        }
        table_3_rows.append(row)

    # Save to table_3.csv
    fieldnames = list(table_3_rows[0].keys())
    with open(TABLE_3_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in table_3_rows:
            writer.writerow(r)

    print(f"\nSuccessfully wrote Table 3 to {TABLE_3_CSV}")
    print("\n------------------ TABLE 3 SUMMARY ------------------")
    for r in table_3_rows:
        print(f"Setting: {r['Setting (piece size, pieces fetched, re-rank yes or no)']}")
        print(f"  Attempted: {r['Questions asked']} | Success: {r['Successfully answered']} | RateLimited: {r['Rate-limited']}")
        print(f"  Correct & Cited: {r['Correct and cited %']} | Wrong: {r['Wrong %']} | Correct Refusal: {r['Correctly refused %']}")
        print(f"  Tokens/Answer: {r['Tokens per answer']} (In: {r['Mean prompt tokens']}, Out: {r['Mean completion tokens']})")
        print(f"  TTFT p50/p95: {r['First token p50 / p95 (ms)']} | Total p50/p95: {r['Total time p50 / p95 (ms)']}\n")

if __name__ == "__main__":
    asyncio.run(run_benchmark())

