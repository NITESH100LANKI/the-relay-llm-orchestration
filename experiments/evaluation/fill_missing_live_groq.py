"""
Fill Missing / Rate-Limited Evaluations in Real Groq Retrieval Benchmark.
Re-executes any question where status != 'ok' with conservative 25s replenishment delays.
Updates submission/retrieval_benchmark_real_groq.json and regenerates submission/tables/table_3.csv.
"""

import sys
import asyncio
import json
import time
import csv
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas.models import AskRequest
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.groq_provider import GroqProvider
from app.evaluation.benchmark_questions import BENCHMARK_QUESTIONS
from app.core.config import settings

BENCHMARK_JSON = settings.base_dir / "submission" / "retrieval_benchmark_real_groq.json"
TABLE_3_CSV = settings.base_dir / "submission" / "tables" / "table_3.csv"

def percentile(vals, p):
    if not vals:
        return 0.0
    return float(np.percentile(vals, p))

async def main():
    if not BENCHMARK_JSON.exists():
        print(f"Error: {BENCHMARK_JSON} does not exist.")
        return

    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    provider = GroqProvider()
    orchestrator = PipelineOrchestrator(provider=provider)

    questions_map = {q["id"]: q for q in BENCHMARK_QUESTIONS}
    rate_limited_indices = [i for i, r in enumerate(records) if r["status"] != "ok"]

    print(f"Found {len(rate_limited_indices)} rate-limited items to re-evaluate:")
    for idx in rate_limited_indices:
        r = records[idx]
        print(f"  - Setting {r['retrieval_setting']} | Question {r['question_id']}")

    for idx in rate_limited_indices:
        r = records[idx]
        s_code = r["retrieval_setting"]
        q_id = r["question_id"]
        q_item = questions_map[q_id]

        print(f"\n[Pacing] Sleeping 25s to ensure token bucket is fully replenished before evaluating ({s_code} | {q_id})...")
        await asyncio.sleep(25.0)

        print(f"Running Setting {s_code} | {q_id}: {q_item['question'][:65]}...")
        req = AskRequest(question=q_item["question"], retrieval_setting=s_code)
        res = await orchestrator.execute(req)

        # Update record
        r["status"] = res.status
        r["answer_grade"] = res.answer_grade
        r["refused"] = res.refused
        r["refusal_reason"] = res.refusal_reason
        r["answer"] = res.answer
        r["citations"] = [
            {
                "paper_title": c.paper_title,
                "arxiv_id": c.arxiv_id,
                "version": c.version,
                "page_number": c.page_number,
                "section": c.section
            }
            for c in res.citations
        ]
        r["retrieved_passages"] = [
            {
                "chunk_id": p.chunk_id,
                "paper_title": p.paper_title,
                "page_number": p.page_number,
                "section": p.section
            }
            for p in res.retrieved_passages
        ]
        r["tokens_in"] = res.tokens_in
        r["tokens_out"] = res.tokens_out
        r["first_token_ms"] = res.first_token_ms
        r["provider_ttft_ms"] = res.provider_ttft_ms
        r["total_ms"] = res.total_ms
        r["retries"] = res.retries
        r["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        r["rate_limit_headers"] = provider.last_rate_limit_headers

        print(f"  Result: Status={res.status} | Grade={res.answer_grade} | Tokens In={res.tokens_in}, Out={res.tokens_out} | TTFT={res.first_token_ms}ms")

        # Save immediately
        with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    # Recompute Table 3
    print("\nRecomputing Table 3 with 100% completed live evaluations...")
    config_meta = [
        ("A", "300 tokens, 3 pieces, re-rank no"),
        ("B", "500 tokens, 5 pieces, re-rank no"),
        ("C", "500 tokens, 5 pieces, re-rank yes")
    ]

    table_3_rows = []
    for s_code, s_label in config_meta:
        s_items = [r for r in records if r["retrieval_setting"] == s_code]
        n_attempted = len(s_items)
        n_ok = len([r for r in s_items if r["status"] == "ok"])
        n_rate_limited = len([r for r in s_items if r["status"] == "rate_limited"])

        n_correct = len([r for r in s_items if r["answer_grade"] == "correct"])
        n_wrong = len([r for r in s_items if r["answer_grade"] == "wrong"])
        n_refused_correctly = len([r for r in s_items if r["answer_grade"] == "refused_correctly"])
        n_refused_wrongly = len([r for r in s_items if r["answer_grade"] == "refused_wrongly"])

        pct_correct = (n_correct / n_attempted) * 100.0
        pct_wrong = (n_wrong / n_attempted) * 100.0
        pct_refused_correctly = (n_refused_correctly / n_attempted) * 100.0

        # Mean tokens per SUCCESSFUL answer
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

    fieldnames = list(table_3_rows[0].keys())
    with open(TABLE_3_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in table_3_rows:
            writer.writerow(r)

    print(f"\nSuccessfully wrote Table 3 to {TABLE_3_CSV}")
    for r in table_3_rows:
        print(f"Setting: {r['Setting (piece size, pieces fetched, re-rank yes or no)']}")
        print(f"  Attempted: {r['Questions asked']} | Success: {r['Successfully answered']} | RateLimited: {r['Rate-limited']}")
        print(f"  Correct & Cited: {r['Correct and cited %']} | Wrong: {r['Wrong %']} | Correct Refusal: {r['Correctly refused %']}")
        print(f"  Tokens/Answer: {r['Tokens per answer']} (In: {r['Mean prompt tokens']}, Out: {r['Mean completion tokens']})")
        print(f"  TTFT p50/p95: {r['First token p50 / p95 (ms)']} | Total p50/p95: {r['Total time p50 / p95 (ms)']}\n")

if __name__ == "__main__":
    asyncio.run(main())
