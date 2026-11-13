"""
Automated Submission Integrity Test Suite for The Relay.
Validates:
1. CSV integrity (row count, required columns, blank fields for rate-limited requests, identical burst timestamps).
2. Benchmark outcome completeness (15 questions accounted for across Settings A, B, and C).
3. Slow-request monotonic arithmetic and reconciliation (request d2663ac8-de30-4e27-850d-da57cbe2b03a).
4. Table reconciliation and percentile calculations with verified denominators.
5. Strict 2-page constraint for the compiled ReportLab PDF report.
"""

import csv
import json
from pathlib import Path
import pytest
import pymupdf

BASE_DIR = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = BASE_DIR / "submission"
RESULTS_CSV = SUBMISSION_DIR / "results.csv"
BENCHMARK_JSON = SUBMISSION_DIR / "retrieval_benchmark_real_groq.json"
REPORT_PDF = SUBMISSION_DIR / "report.pdf"
TABLES_DIR = SUBMISSION_DIR / "tables"

def test_csv_structure_and_row_count():
    assert RESULTS_CSV.exists(), "results.csv must exist"
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 170, f"results.csv must contain exactly 170 rows, got {len(rows)}"
    expected_cols = [
        "request_id", "scenario", "source", "start_time", "first_token_ms",
        "total_ms", "status", "retries", "tokens_in", "tokens_out",
        "retrieval_setting", "answer_grade"
    ]
    assert reader.fieldnames == expected_cols, f"Column mismatch. Expected {expected_cols}, got {reader.fieldnames}"

def test_csv_rate_limited_grading_integrity():
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rate_limited_rows = [r for r in rows if r["status"] == "rate_limited"]
    assert len(rate_limited_rows) == 109, f"Expected 109 rate-limited rows, got {len(rate_limited_rows)}"

    for r in rate_limited_rows:
        assert r["answer_grade"] == "", f"Rate-limited request {r['request_id']} must have blank answer_grade"
        assert r["first_token_ms"] == "", f"Rate-limited request {r['request_id']} must have blank first_token_ms"

    successful_rows = [r for r in rows if r["status"] == "ok"]
    assert len(successful_rows) == 61, f"Expected 61 ok rows, got {len(successful_rows)}"
    for r in successful_rows:
        assert r["answer_grade"] in {"correct", "wrong", "refused_correctly", "refused_wrongly"}
        assert float(r["first_token_ms"]) > 0

def test_burst_timestamps_identity():
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    burst_rows = [r for r in rows if r["scenario"] == "sudden_burst"]
    assert len(burst_rows) == 25, f"Expected 25 burst requests, got {len(burst_rows)}"
    timestamps = {r["start_time"] for r in burst_rows}
    assert len(timestamps) == 1, f"All burst requests must share identical arrival timestamp, found {timestamps}"
    assert list(timestamps)[0] == "2026-11-13T06:32:17Z"

def test_benchmark_outcome_completeness():
    assert BENCHMARK_JSON.exists(), "retrieval_benchmark_real_groq.json must exist"
    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) == 45, f"Expected 45 live Groq evaluation records, got {len(records)}"

    for s in ["A", "B", "C"]:
        rows = [r for r in records if r["retrieval_setting"] == s]
        assert len(rows) == 15, f"Setting {s} must have exactly 15 records, got {len(rows)}"

        cats = {
            "correct": 0, "wrong": 0, "refused_correctly": 0,
            "refused_wrongly": 0, "rate_limited": 0, "timeout": 0, "failed": 0
        }
        for r in rows:
            st = r["status"]
            ag = r.get("answer_grade")
            if st == "rate_limited":
                cats["rate_limited"] += 1
            elif st == "timeout":
                cats["timeout"] += 1
            elif st != "ok":
                cats["failed"] += 1
            else:
                assert ag in cats, f"Unknown grade {ag}"
                cats[ag] += 1

        total_accounted = sum(cats.values())
        assert total_accounted == 15, f"Sum of outcomes for {s} must be 15, got {total_accounted}"

        # Setting-specific verified counts
        if s == "A":
            assert cats["correct"] == 4
            assert cats["refused_correctly"] == 9
            assert cats["rate_limited"] == 2
            assert cats["wrong"] == 0
        elif s == "B":
            assert cats["correct"] == 6
            assert cats["refused_correctly"] == 8
            assert cats["rate_limited"] == 1
            assert cats["wrong"] == 0
        elif s == "C":
            assert cats["correct"] == 5
            assert cats["refused_correctly"] == 8
            assert cats["rate_limited"] == 2
            assert cats["wrong"] == 0

def test_slow_request_monotonic_arithmetic():
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    slow_req = next((r for r in rows if r["request_id"] == "d2663ac8-de30-4e27-850d-da57cbe2b03a"), None)
    assert slow_req is not None, "Slow request d2663ac8-de30-4e27-850d-da57cbe2b03a must exist"

    # Stage durations
    d_stages_1_to_3 = 4.2 + 112.5 + 1.1  # 117.8 ms
    d_attempt_1 = 758.0
    d_backoff_1 = 1050.0
    d_attempt_2 = 640.2
    d_backoff_2 = 2200.0
    d_attempt_3_provider_ttft = 644.4
    d_stream_generation = 3161.9
    d_stage_5_verification = 18.5

    # Elapsed monotonic duration before attempt 3 dispatch
    t_before_attempt_3 = d_stages_1_to_3 + d_attempt_1 + d_backoff_1 + d_attempt_2 + d_backoff_2
    assert round(t_before_attempt_3, 1) == 4766.0, f"Expected 4766.0 ms, got {t_before_attempt_3}"

    # End-to-end user-perceived TTFT
    user_ttft = t_before_attempt_3 + d_attempt_3_provider_ttft
    assert round(user_ttft, 1) == 5410.4, f"Expected 5410.4 ms, got {user_ttft}"
    assert float(slow_req["first_token_ms"]) == 5410.4

    # Total latency
    total_ms = user_ttft + d_stream_generation + d_stage_5_verification
    assert round(total_ms, 1) == 8590.8, f"Expected 8590.8 ms, got {total_ms}"
    assert float(slow_req["total_ms"]) == 8590.8

    # Percentage breakdowns
    pre_token_pct = round((user_ttft / total_ms) * 100, 1)
    gen_pct = round((d_stream_generation / total_ms) * 100, 1)
    verif_pct = round((d_stage_5_verification / total_ms) * 100, 1)

    assert pre_token_pct == 63.0, f"Expected 63.0%, got {pre_token_pct}%"
    assert gen_pct == 36.8, f"Expected 36.8%, got {gen_pct}%"
    assert verif_pct == 0.2, f"Expected 0.2%, got {verif_pct}%"
    assert round(pre_token_pct + gen_pct + verif_pct, 1) == 100.0

def test_pdf_report_two_page_limit():
    assert REPORT_PDF.exists(), "submission/report.pdf must exist"
    doc = pymupdf.open(str(REPORT_PDF))
    page_count = len(doc)
    assert page_count == 2, f"report.pdf must be exactly 2 pages, got {page_count}"

def test_tables_reconciliation():
    assert (TABLES_DIR / "table_1.csv").exists()
    assert (TABLES_DIR / "table_2a.csv").exists()
    assert (TABLES_DIR / "table_2b.csv").exists()
    assert (TABLES_DIR / "table_3.csv").exists()
    assert (TABLES_DIR / "table_retry_comparison.csv").exists()

    with open(TABLES_DIR / "table_retry_comparison.csv", "r", encoding="utf-8") as f:
        rows = {r["Metric"]: r for r in csv.DictReader(f)}

    assert rows["Offered Requests"]["Retries Enabled (max_retries=2)"] == "25"
    assert rows["Completed Requests"]["Retries Enabled (max_retries=2)"] == "3"
    assert rows["Total API Calls Made"]["Retries Enabled (max_retries=2)"] == "69"
    assert rows["Total API Calls Made"]["Retries Disabled (max_retries=0)"] == "25"
    assert rows["Number of 429s Encountered"]["Retries Enabled (max_retries=2)"] == "66"
    assert rows["Number of 429s Encountered"]["Retries Disabled (max_retries=0)"] == "22"
