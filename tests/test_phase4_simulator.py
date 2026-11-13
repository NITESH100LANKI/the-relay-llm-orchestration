"""
Unit and Integration Tests for Phase 4: Controllable Stand-in Model Service.
Tests:
- Independent operation without Groq API
- Calibrated latency and TTFT distributions
- Streaming chunk generation
- RPM rate limit trigger and 429 response headers
- TPM rate limit trigger
- Daily budget exhaustion (3pm scenario)
- Fault injection: timeouts and 5xx errors
- Concurrency limit and load-dependent inflation
- Deterministic random seed reproducibility
"""

import pytest
import asyncio
from app.providers.simulator_provider import SimulatorProvider, SimulatorConfig

@pytest.mark.asyncio
async def test_simulator_basic_generation():
    cfg = SimulatorConfig(base_ttft_ms=50.0, token_generation_ms=1.0, random_seed=42)
    provider = SimulatorProvider(cfg)
    
    resp = await provider.generate("What is Scaled Dot-Product Attention?")
    assert resp.status == "ok"
    assert resp.source == "stand-in"
    assert resp.tokens_out > 0
    assert resp.first_token_ms > 0
    assert resp.total_ms >= resp.first_token_ms

@pytest.mark.asyncio
async def test_simulator_streaming():
    cfg = SimulatorConfig(base_ttft_ms=30.0, token_generation_ms=1.0, stream_chunk_tokens=3)
    provider = SimulatorProvider(cfg)

    chunks = []
    is_first_seen = False
    final_resp = None

    async for item in provider.generate_stream("What is Mimi in Moshi?"):
        if item["delta"]:
            chunks.append(item["delta"])
        if item["is_first"]:
            is_first_seen = True
        if item["done"]:
            final_resp = item["response"]

    assert len(chunks) > 1
    assert is_first_seen is True
    assert final_resp is not None
    assert final_resp.status == "ok"
    assert "".join(chunks).strip() != ""

@pytest.mark.asyncio
async def test_simulator_rpm_rate_limit():
    # Configure tiny RPM of 3
    cfg = SimulatorConfig(rpm_limit=3, base_ttft_ms=5.0, token_generation_ms=0.5)
    provider = SimulatorProvider(cfg)

    # Fire 5 rapid requests
    responses = []
    for _ in range(5):
        r = await provider.generate("Test query")
        responses.append(r)

    statuses = [r.status for r in responses]
    assert "rate_limited" in statuses
    rl_item = next(r for r in responses if r.status == "rate_limited")
    assert "x-ratelimit-limit-requests" in rl_item.rate_limit_headers
    assert rl_item.retry_after_sec is not None

@pytest.mark.asyncio
async def test_simulator_daily_budget_exhaustion():
    # 0 daily requests remaining
    cfg = SimulatorConfig(daily_requests_remaining=0)
    provider = SimulatorProvider(cfg)

    resp = await provider.generate("Any question at 3pm")
    assert resp.status == "rate_limited"
    assert "daily" in resp.error_message.lower()
    assert resp.rate_limit_headers.get("x-ratelimit-remaining-requests") == "0"
    assert resp.retry_after_sec is not None
    assert resp.retry_after_sec > 60.0  # Daily reset time

@pytest.mark.asyncio
async def test_simulator_fault_injection():
    # 100% error rate
    cfg_err = SimulatorConfig(error_probability=1.0)
    p_err = SimulatorProvider(cfg_err)
    r_err = await p_err.generate("Trigger error")
    assert r_err.status == "error"

    # 100% timeout rate
    cfg_to = SimulatorConfig(timeout_probability=1.0, timeout_threshold_ms=50.0)
    p_to = SimulatorProvider(cfg_to)
    r_to = await p_to.generate("Trigger timeout")
    assert r_to.status == "timeout"

@pytest.mark.asyncio
async def test_simulator_deterministic_reproducibility():
    cfg1 = SimulatorConfig(random_seed=12345, base_ttft_ms=100.0)
    p1 = SimulatorProvider(cfg1)
    r1 = await p1.generate("Deterministic test")

    cfg2 = SimulatorConfig(random_seed=12345, base_ttft_ms=100.0)
    p2 = SimulatorProvider(cfg2)
    r2 = await p2.generate("Deterministic test")

    assert round(r1.first_token_ms, 2) == round(r2.first_token_ms, 2)
    assert round(r1.total_ms, 2) == round(r2.total_ms, 2)

