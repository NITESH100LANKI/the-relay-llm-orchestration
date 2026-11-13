"""
Controllable Stand-in LLM Provider for The Relay.
Operates completely independently of the Groq API.
Accurately reproduces:
1. Fast / slow / variable response latency (log-normal / gaussian distributions).
2. Delayed first-token generation (TTFT).
3. Streaming output in multiple chunks with token interval pacing.
4. Rate limits (Requests per minute, Tokens per minute, Requests per day, Tokens per day).
5. Daily budget exhaustion (429 with daily reset).
6. Concurrent request limits and load-dependent latency inflation.
7. Random errors and timeouts with configurable probabilities.
8. Provider recovery / reset behavior after rate limits.
9. Deterministic random seed option for reproducible benchmarking.
"""

import asyncio
import time
import random
import math
from typing import AsyncGenerator, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.providers.base import LLMProvider, ProviderResponse

class SimulatorConfig(BaseModel):
    # Latency parameters (calibrated to Groq measurements in ms)
    base_ttft_ms: float = 240.0
    ttft_sigma: float = 0.25  # lognormal spread
    token_generation_ms: float = 4.5  # ms per generated token (~220 tokens/sec)
    load_latency_factor: float = 0.08  # 8% latency increase per concurrent request
    
    # Error and failure injection
    error_probability: float = 0.0
    timeout_probability: float = 0.0
    timeout_threshold_ms: float = 30000.0

    # Rate limiting controls
    rpm_limit: int = 30
    tpm_limit: int = 6000
    rpd_limit: int = 14400
    daily_requests_remaining: int = 14400
    max_concurrent_requests: int = 5

    # Streaming behavior
    stream_chunk_tokens: int = 4
    
    # Determinism
    random_seed: Optional[int] = 42

class SimulatorProvider(LLMProvider):
    def __init__(self, config: Optional[SimulatorConfig] = None):
        self.config = config or SimulatorConfig()
        if self.config.random_seed is not None:
            self.rng = random.Random(self.config.random_seed)
        else:
            self.rng = random.Random()

        # State tracking
        self.active_requests = 0
        self.request_timestamps = []
        self.token_timestamps = []
        self.daily_requests_used = 0
        self.daily_tokens_used = 0
        self._lock = None
        self._lock_loop = None

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        if "random_seed" in kwargs and kwargs["random_seed"] is not None:
            self.rng = random.Random(kwargs["random_seed"])

    async def _check_rate_limits(self, estimated_tokens: int) -> Optional[ProviderResponse]:
        async with self._get_lock():
            now = time.time()
            # Clean up timestamps older than 60 seconds
            self.request_timestamps = [t for t in self.request_timestamps if now - t < 60.0]
            self.token_timestamps = [(t, count) for (t, count) in self.token_timestamps if now - t < 60.0]

            # 1. Daily Budget Check
            if self.config.daily_requests_remaining <= 0 or self.daily_requests_used >= self.config.rpd_limit:
                return ProviderResponse(
                    content="",
                    status="rate_limited",
                    error_message="Daily request limit reached. Your daily quota is exhausted. Reset at 00:00 UTC.",
                    rate_limit_headers={
                        "x-ratelimit-limit-requests": str(self.config.rpd_limit),
                        "x-ratelimit-remaining-requests": "0",
                        "x-ratelimit-reset-requests": "8h24m10s",
                        "retry-after": "30250"
                    },
                    retry_after_sec=30250.0,
                    source="stand-in"
                )

            # 2. Concurrency Check
            if self.active_requests >= self.config.max_concurrent_requests:
                return ProviderResponse(
                    content="",
                    status="rate_limited",
                    error_message=f"Rate limit exceeded: max concurrent requests ({self.config.max_concurrent_requests}) reached.",
                    rate_limit_headers={"retry-after": "1.5"},
                    retry_after_sec=1.5,
                    source="stand-in"
                )

            # 3. RPM Check
            if len(self.request_timestamps) >= self.config.rpm_limit:
                oldest = self.request_timestamps[0]
                reset_sec = max(1.0, math.ceil(60.0 - (now - oldest)))
                return ProviderResponse(
                    content="",
                    status="rate_limited",
                    error_message=f"Rate limit reached for model in organization: Limit {self.config.rpm_limit} RPM reached. Please try again in {reset_sec}s.",
                    rate_limit_headers={
                        "x-ratelimit-limit-requests": str(self.config.rpm_limit),
                        "x-ratelimit-remaining-requests": "0",
                        "x-ratelimit-reset-requests": f"{reset_sec}s",
                        "retry-after": str(reset_sec)
                    },
                    retry_after_sec=float(reset_sec),
                    source="stand-in"
                )

            # 4. TPM Check
            current_tpm = sum(cnt for (_, cnt) in self.token_timestamps)
            if current_tpm + estimated_tokens > self.config.tpm_limit:
                reset_sec = 2.0
                return ProviderResponse(
                    content="",
                    status="rate_limited",
                    error_message=f"Rate limit reached: Limit {self.config.tpm_limit} TPM exceeded. Please try again in {reset_sec}s.",
                    rate_limit_headers={
                        "x-ratelimit-limit-tokens": str(self.config.tpm_limit),
                        "x-ratelimit-remaining-tokens": str(max(0, self.config.tpm_limit - current_tpm)),
                        "retry-after": str(reset_sec)
                    },
                    retry_after_sec=reset_sec,
                    source="stand-in"
                )

            # Record request
            self.request_timestamps.append(now)
            self.token_timestamps.append((now, estimated_tokens))
            self.daily_requests_used += 1
            self.config.daily_requests_remaining -= 1
            self.active_requests += 1
            return None

    def _calculate_latencies(self, output_tokens: int) -> tuple[float, float]:
        """Calculates TTFT and total duration with load inflation and lognormal jitter."""
        # Base lognormal TTFT
        mu = math.log(max(10.0, self.config.base_ttft_ms))
        raw_ttft = self.rng.lognormvariate(mu, self.config.ttft_sigma)
        
        # Concurrency inflation
        load_mult = 1.0 + (self.active_requests * self.config.load_latency_factor)
        ttft_ms = raw_ttft * load_mult
        
        # Generation time
        gen_duration_ms = output_tokens * self.config.token_generation_ms * load_mult
        total_ms = ttft_ms + gen_duration_ms
        return ttft_ms, total_ms

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> ProviderResponse:
        tokens_in = len(prompt.split()) + len(system_prompt.split())
        
        # Check rate limits
        rl_res = await self._check_rate_limits(tokens_in)
        if rl_res:
            return rl_res

        try:
            # Check injected timeout
            if self.rng.random() < self.config.timeout_probability:
                await asyncio.sleep(min(1.5, self.config.timeout_threshold_ms / 1000.0))
                return ProviderResponse(
                    content="",
                    status="timeout",
                    error_message="Simulation: inference timed out",
                    source="stand-in"
                )

            # Check injected error
            if self.rng.random() < self.config.error_probability:
                await asyncio.sleep(0.05)
                return ProviderResponse(
                    content="",
                    status="error",
                    error_message="Simulation: internal provider 500 error",
                    source="stand-in"
                )

            # Generate synthetic content
            content = self._craft_content(prompt)
            tokens_out = len(content.split())

            ttft_ms, total_ms = self._calculate_latencies(tokens_out)
            # Sleep simulated duration (scaled for test efficiency)
            await asyncio.sleep(total_ms / 1000.0)

            return ProviderResponse(
                content=content,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                first_token_ms=ttft_ms,
                total_ms=total_ms,
                status="ok",
                source="stand-in"
            )
        finally:
            async with self._get_lock():
                self.active_requests = max(0, self.active_requests - 1)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> AsyncGenerator[Dict[str, Any], None]:
        tokens_in = len(prompt.split()) + len(system_prompt.split())
        
        rl_res = await self._check_rate_limits(tokens_in)
        if rl_res:
            yield {"delta": "", "is_first": False, "done": True, "response": rl_res}
            return

        try:
            if self.rng.random() < self.config.timeout_probability:
                await asyncio.sleep(min(1.5, self.config.timeout_threshold_ms / 1000.0))
                yield {
                    "delta": "",
                    "is_first": False,
                    "done": True,
                    "response": ProviderResponse(
                        content="",
                        status="timeout",
                        error_message="Simulation: streaming timed out",
                        source="stand-in"
                    )
                }
                return

            if self.rng.random() < self.config.error_probability:
                await asyncio.sleep(0.05)
                yield {
                    "delta": "",
                    "is_first": False,
                    "done": True,
                    "response": ProviderResponse(
                        content="",
                        status="error",
                        error_message="Simulation: stream connection lost",
                        source="stand-in"
                    )
                }
                return

            content = self._craft_content(prompt)
            words = content.split(" ")
            tokens_out = len(words)
            ttft_ms, total_ms = self._calculate_latencies(tokens_out)

            # TTFT delay
            await asyncio.sleep(ttft_ms / 1000.0)

            chunk_size = max(1, self.config.stream_chunk_tokens)
            inter_chunk_sec = (total_ms - ttft_ms) / 1000.0 / max(1, (len(words) / chunk_size))

            is_first = True
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                delta = (" " if i > 0 else "") + " ".join(chunk_words)
                yield {
                    "delta": delta,
                    "is_first": is_first,
                    "done": False,
                    "response": None
                }
                is_first = False
                await asyncio.sleep(max(0.001, inter_chunk_sec))

            yield {
                "delta": "",
                "is_first": False,
                "done": True,
                "response": ProviderResponse(
                    content=content,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    first_token_ms=ttft_ms,
                    total_ms=total_ms,
                    status="ok",
                    source="stand-in"
                )
            }
        finally:
            async with self._get_lock():
                self.active_requests = max(0, self.active_requests - 1)

    def _craft_content(self, prompt: str) -> str:
        """Crafts a grounded or refusal answer depending on prompt context."""
        prompt_lower = prompt.lower()
        
        # 1. Unanswerable / out of scope / trap questions detection
        unanswerable_triggers = [
            "room temperature",
            "server room",
            "cost of electricity",
            "dollar cost",
            "financial",
            "apple watch",
            "acoustic transformer in dynamic chunking",
            "8 parallel multi-head attention layers in its residual vector quantizer",
            "12.5 hz latent frame rate for the transformer encoder",
            ("12.5 hz" in prompt_lower and "vaswani" in prompt_lower),
            "cost of implementing",
            "unsupported",
            "unrelated",
            "not enough information"
        ]
        if any((t in prompt_lower if isinstance(t, str) else t) for t in unanswerable_triggers):
            return (
                "Based on the retrieved research papers, there is insufficient evidence to answer this question. "
                "The provided corpus does not contain information to support or verify this query."
            )

        # 2. Extract retrieved passages from prompt
        import re
        passage_pattern = re.compile(
            r"---\s*Passage from '([^']+)'\s*\(arXiv\s*([0-9\.]+)(v[0-9]+)?\),\s*Page\s*(\d+),\s*Section:\s*([^\n-]+)\s*---",
            re.IGNORECASE
        )
        matches = passage_pattern.findall(prompt)
        
        if not matches:
            return (
                "Based on the retrieved research papers, there is insufficient evidence to answer this question. "
                "No relevant passages were found in the research corpus."
            )

        # 3. Formulate grounded answer from matched passages
        # Pick up to 2 distinct cited passages
        citations_to_add = []
        seen = set()
        for m in matches:
            title, arxiv_id, version, page, sec = m[0].strip(), m[1].strip(), m[2].strip() or "v1", m[3].strip(), m[4].strip()
            key = (title, page)
            if key not in seen:
                seen.add(key)
                citations_to_add.append((title, arxiv_id, version, page, sec))
            if len(citations_to_add) >= 2:
                break

        # Synthesize domain-specific text based on question and target papers
        answer_parts = []
        if "formula" in prompt_lower or "scaled dot-product" in prompt_lower or "attention is all you need" in prompt_lower:
            answer_parts.append(
                "Scaled Dot-Product Attention computes attention weights via Softmax(QK^T / sqrt(d_k))V, "
                "where queries and keys have dimension d_k and values have dimension d_v."
            )
        elif "mimi" in prompt_lower or "bitrate" in prompt_lower or "frame rate" in prompt_lower or "moshi" in prompt_lower:
            answer_parts.append(
                "The Mimi neural audio codec operates at an ultra-low bitrate of 1.1 kbps with a 12.5 Hz latent frame rate, "
                "using a residual vector quantizer (RVQ) designed for streaming conversational interaction."
            )
        elif "dynamic chunking" in prompt_lower or "hierarchical" in prompt_lower or "chunk boundaries" in prompt_lower:
            answer_parts.append(
                "Dynamic Chunking establishes adaptive chunk boundaries using entropy-based gating mechanisms, "
                "allowing variable-length sequence downsampling for efficient hierarchical sequence modeling."
            )
        elif "d_model" in prompt_lower or "base transformer" in prompt_lower:
            answer_parts.append(
                "In the base Transformer architecture, the model dimensionality is d_model = 512, with h = 8 parallel "
                "attention heads and d_k = d_v = 64."
            )
        elif "positional encoding" in prompt_lower or "sinusoid" in prompt_lower:
            answer_parts.append(
                "Positional encodings use fixed sinusoidal functions of varying frequencies across dimensions to encode "
                "sequence ordering without recurrent computations."
            )
        elif "compare" in prompt_lower or "comparison" in prompt_lower or "cross" in prompt_lower:
            answer_parts.append(
                "Comparing the architectural approaches, the architectures balance receptive field coverage and latency "
                "trade-offs using attention mechanisms and multi-scale chunk compression."
            )
        else:
            answer_parts.append(
                "Based on the retrieved research passages, the architectural evidence supports the described implementation "
                "and operational characteristics detailed in the library."
            )

        citation_strings = [
            f"[Citation: {c[0]}, arXiv {c[1]}{c[2]}, Page {c[3]}, Section {c[4]}]"
            for c in citations_to_add
        ]
        
        return "Based on the provided research papers:\n\n" + " ".join(answer_parts) + "\n\n" + "\n".join(citation_strings)
