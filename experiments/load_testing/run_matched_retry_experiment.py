"""
Matched Retry Policy Comparison Experiment under Overload Conditions.
Authoritative Requirement 8:
Runs a matched empirical experiment under identical traffic conditions:
- 25 requests under sustained overload (RPM=15, max_concurrent=3, seed=606)
- Run A: Retries Enabled (max_retries = 2)
- Run B: Retries Disabled (max_retries = 0)

Measures and records:
- Completed requests
- Failed requests
- P50 / P95 latency of completed requests
- Total API calls made
- Number of 429s encountered
- Peak queue length or concurrency

Outputs side-by-side comparison to submission/tables/table_retry_comparison.csv.
"""

import sys
import asyncio
import csv
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas.models import AskRequest, PipelineResult
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.simulator_provider import SimulatorProvider, SimulatorConfig
from app.evaluation.benchmark_questions import BENCHMARK_QUESTIONS
from app.core.reliability import ReliabilityManager
from app.core.config import settings

OUTPUT_CSV = settings.base_dir / "submission" / "tables" / "table_retry_comparison.csv"

def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(vals, p))

async def run_overload_suite(max_retries: int, seed: int = 606, request_count: int = 25) -> Dict[str, Any]:
    # Configure deterministic simulator under overload
    sim_cfg = SimulatorConfig(
        rpm_limit=15,
        tpm_limit=4000,
        max_concurrent_requests=3,
        random_seed=seed
    )
    provider = SimulatorProvider(sim_cfg)
    
    # Fresh isolated reliability manager for clean queue & concurrency tracking
    rel_mgr = ReliabilityManager(max_concurrency=3, rpm_limit=15)
    
    # Patch orchestrator and core reliability to use this isolated reliability manager
    import app.pipeline.orchestrator as orch_module
    import app.core.reliability as rel_module
    old_rel_mgr = orch_module.reliability_mgr
    orch_module.reliability_mgr = rel_mgr
    rel_module.reliability_mgr = rel_mgr

    orchestrator = PipelineOrchestrator(provider=provider, max_retries=max_retries)

    results: List[PipelineResult] = []

    async def send_request(idx: int):
        q_item = BENCHMARK_QUESTIONS[idx % len(BENCHMARK_QUESTIONS)]
        req = AskRequest(question=q_item["question"], retrieval_setting="B")
        res = await orchestrator.execute(req)
        results.append(res)
        return res

    # Launch requests with 0.8s pacing (75 req/min offered against 15 RPM limit = 5x overload)
    tasks = []
    for i in range(request_count):
        tasks.append(asyncio.create_task(send_request(i)))
        await asyncio.sleep(0.8)

    await asyncio.gather(*tasks)

    # Restore reliability manager
    orch_module.reliability_mgr = old_rel_mgr
    rel_module.reliability_mgr = old_rel_mgr

    # Compute metrics
    completed = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]

    completed_latencies = [r.total_ms for r in completed]
    p50_latency = percentile(completed_latencies, 50)
    p95_latency = percentile(completed_latencies, 95)

    # In our pipeline: every attempt made is 1 initial attempt + retries count
    total_api_calls = sum(1 + r.retries for r in results)
    
    # 429 encounters: each retry was prompted by a 429, plus failed requests that ended with 429
    rate_limit_429s = sum(r.retries + (1 if r.status == "rate_limited" else 0) for r in results)

    return {
        "policy": f"Retries Enabled (max_retries={max_retries})" if max_retries > 0 else "Retries Disabled (max_retries=0)",
        "max_retries": max_retries,
        "offered_requests": request_count,
        "completed_requests": len(completed),
        "failed_requests": len(failed),
        "p50_latency_ms": round(p50_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "total_api_calls": total_api_calls,
        "rate_limit_429_count": rate_limit_429s,
        "peak_queue_length": rel_mgr.peak_queue_length
    }

async def main():
    print("=" * 70, flush=True)
    print("RUNNING MATCHED RETRY POLICY COMPARISON EXPERIMENT", flush=True)
    print("Traffic: 25 requests at 5x overload (75 RPM offered vs 15 RPM capacity)", flush=True)
    print("=" * 70, flush=True)

    print("\n>>> Run A: Retries Enabled (max_retries=2)...", flush=True)
    run_a = await run_overload_suite(max_retries=2, seed=606, request_count=25)
    print(f"Run A completed: {run_a['completed_requests']} ok, {run_a['failed_requests']} failed", flush=True)

    # Brief cooldown
    await asyncio.sleep(2.0)

    print("\n>>> Run B: Retries Disabled (max_retries=0)...", flush=True)
    run_b = await run_overload_suite(max_retries=0, seed=606, request_count=25)
    print(f"Run B completed: {run_b['completed_requests']} ok, {run_b['failed_requests']} failed", flush=True)

    # Format side-by-side comparison table
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    rows = [
        {
            "Metric": "Offered Requests",
            "Retries Enabled (max_retries=2)": run_a["offered_requests"],
            "Retries Disabled (max_retries=0)": run_b["offered_requests"],
            "Delta / Impact": "Identical traffic load (seed=606)"
        },
        {
            "Metric": "Completed Requests",
            "Retries Enabled (max_retries=2)": run_a["completed_requests"],
            "Retries Disabled (max_retries=0)": run_b["completed_requests"],
            "Delta / Impact": f"{run_a['completed_requests'] - run_b['completed_requests']:+d} completed"
        },
        {
            "Metric": "Failed Requests",
            "Retries Enabled (max_retries=2)": run_a["failed_requests"],
            "Retries Disabled (max_retries=0)": run_b["failed_requests"],
            "Delta / Impact": f"{run_a['failed_requests'] - run_b['failed_requests']:+d} failures"
        },
        {
            "Metric": "Completed Latency p50 (ms)",
            "Retries Enabled (max_retries=2)": run_a["p50_latency_ms"],
            "Retries Disabled (max_retries=0)": run_b["p50_latency_ms"],
            "Delta / Impact": f"{run_a['p50_latency_ms'] - run_b['p50_latency_ms']:+.1f} ms"
        },
        {
            "Metric": "Completed Latency p95 (ms)",
            "Retries Enabled (max_retries=2)": run_a["p95_latency_ms"],
            "Retries Disabled (max_retries=0)": run_b["p95_latency_ms"],
            "Delta / Impact": f"{run_a['p95_latency_ms'] - run_b['p95_latency_ms']:+.1f} ms"
        },
        {
            "Metric": "Total API Calls Made",
            "Retries Enabled (max_retries=2)": run_a["total_api_calls"],
            "Retries Disabled (max_retries=0)": run_b["total_api_calls"],
            "Delta / Impact": f"{run_a['total_api_calls'] - run_b['total_api_calls']:+d} calls ({((run_a['total_api_calls']/max(1, run_b['total_api_calls']))-1)*100:+.1f}%)"
        },
        {
            "Metric": "Number of 429s Encountered",
            "Retries Enabled (max_retries=2)": run_a["rate_limit_429_count"],
            "Retries Disabled (max_retries=0)": run_b["rate_limit_429_count"],
            "Delta / Impact": f"{run_a['rate_limit_429_count'] - run_b['rate_limit_429_count']:+d} HTTP 429s"
        },
        {
            "Metric": "Peak Queue Length",
            "Retries Enabled (max_retries=2)": run_a["peak_queue_length"],
            "Retries Disabled (max_retries=0)": run_b["peak_queue_length"],
            "Delta / Impact": f"{run_a['peak_queue_length'] - run_b['peak_queue_length']:+d} waiting requests"
        }
    ]

    fieldnames = ["Metric", "Retries Enabled (max_retries=2)", "Retries Disabled (max_retries=0)", "Delta / Impact"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSuccessfully saved Matched Retry Comparison to {OUTPUT_CSV}")
    print("\n" + "=" * 70)
    print("MATCHED RETRY POLICY COMPARISON TABLE")
    print("=" * 70)
    for r in rows:
        print(f"{r['Metric']:<30} | {str(r['Retries Enabled (max_retries=2)']):<15} | {str(r['Retries Disabled (max_retries=0)']):<15} | {r['Delta / Impact']}")

if __name__ == "__main__":
    asyncio.run(main())
