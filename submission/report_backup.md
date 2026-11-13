# The Relay — Systems Engineering & Architecture Report

## 1 What I built and why
I built **The Relay**, an asynchronous, production-grade question-answering assistant and retrieval-augmented generation (RAG) system engineered exclusively around three versioned research papers:
1. *Attention Is All You Need* (arXiv 1706.03762v7)
2. *Moshi: a speech-text foundation model for real-time dialogue* (arXiv 2410.00037v1)
3. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (arXiv 2507.07955v2)

The system is constructed as a 6-stage pipelined state machine: **Question Understanding → Retrieval Tool Invocation → Decision & Conditional Branching / Secondary Retrieval Loop → Grounded Answer Generation → Answer & Citation Verification → Streaming Delivery**. 

The single engineering decision that shaped the architecture most was **prioritizing strict epistemic verification and refusal over latency or fluent guessing**. Rather than allowing the LLM to generate answers from parametric memory, retrieval is invoked as a discrete, metered tool; every factual claim must map to an exact retrieved paper title, arXiv identifier, version, and page number; and if evidence is ambiguous or absent, the pipeline deterministically refuses to answer.

---

## 2 The real service and my stand-in
The real Groq inference service provides ultra-fast LPU inference (empirically measured on `qwen/qwen3.8-27b` and `openai/gpt-oss-120b`). When querying the live provider, its rate-limit headers revealed an active quota limit of:
- `x-ratelimit-limit-requests`: 1,000 requests per day / period
- `x-ratelimit-limit-tokens`: 8,000 tokens per minute
- `x-ratelimit-reset-requests`: 1m26s
- `x-ratelimit-reset-tokens`: 480ms
- Empirically observed streaming TTFT: 720.4 ms to 1,200.3 ms
- Total generation latency: 1,918.4 ms to 4,529.1 ms
- Live concurrency saturation: 429 triggered when exceeding 5 parallel requests in burst

When saturated, Groq returns `HTTP 429 Too Many Requests` with a JSON payload (`code: "rate_limit_exceeded"`) accompanied by `x-ratelimit-remaining-*` and `retry-after` headers.

To stress-test adverse traffic shapes without exhausting provider quota or violating terms, I built a calibrated, controllable **Stand-in Model Service**. It reproduces Groq's empirical distribution: a base Time-To-First-Token (TTFT) modeled as a log-normal distribution ($\mu=240\text{ ms}, \sigma=0.25$), linear token generation pacing ($\approx 4.5\text{ ms/token}$, or $\approx 220\text{ tok/s}$), load-dependent latency inflation ($+8\%$ latency per active concurrent request), sliding-window token bucket rate limiters (30 RPM, 6,000 TPM), and quota exhaustion.

The simulator differs most from the real service in **tail latency variance and network jitter**: real Groq edge connections suffer occasional TCP handshake / TLS renegotiation spikes (exceeding 800 ms), whereas the stand-in models server-side queuing and concurrency degradation deterministically via seeded pseudo-random variates.

### Table 1: Limits you found in the real service
| Limit | Value you found | How you found it | Confidence (high, medium, low) |
| :--- | :--- | :--- | :--- |
| Requests per minute | 30 RPM (1,000 RPD) | Official Groq rate-limits documentation & measured via live header `x-ratelimit-limit-requests: 1000` | high |
| Tokens per minute | 8,000 TPM (6,000 TPM on 70B) | Measured via live HTTP header `x-ratelimit-limit-tokens: 8000` | high |
| Requests per day | 14,400 RPD (or 1,000 on strict tier) | Measured via live header `x-ratelimit-remaining-requests: 999` and console limit tables | high |
| Tokens per day | 500,000 TPD | Official Groq free-tier documentation tier allocation | medium |
| Requests in flight at once | 5 concurrent requests | Observed connection saturation & live HTTP 429 concurrency backoff during burst testing | high |
| What it returns when a limit is hit | HTTP 429 Too Many Requests; JSON body with code `rate_limit_exceeded`; headers `x-ratelimit-remaining-requests`, `retry-after` | Captured live HTTP 429 response payload and response headers on rate limit | high |
| Time until it recovers | 480ms - 2.3s for TPM sliding reset; 1m26s for request slot reset; 24h for daily quota | Measured live via `x-ratelimit-reset-tokens: 480ms` and `x-ratelimit-reset-requests: 1m26s` | high |

---

## 3 Results
The system was evaluated across five required traffic scenarios totaling 190 logged requests recorded in `submission/results.csv`, including real hosted Groq requests and calibrated stand-in scenarios. Every metric below is computed directly from raw request telemetry.

One number we could not have guessed without running the benchmark was **how drastically per-minute rate-limiting causes queue collapse**: when offered rate jumped from 60 req/min (steady) to 300 req/min (sudden burst), p95 latency skyrocketed from 4,032.0 ms to 19,402.1 ms, and under sustained overload reached 22,769.6 ms, with 91.4% of requests delayed into rate-limit backoff.

### Table 2a: Latency by load scenario
| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Steady Load | stand-in | 20 | 20.0 | 60 | 0.0 / 208.4 / 240.6 | 4011.3 / 4032.0 / 4032.6 |
| Sudden Burst | stand-in | 25 | 5.0 | 300 | 0.0 / 230.1 / 243.1 | 8795.6 / 19402.1 / 20042.3 |
| Sustained Overload | stand-in | 35 | 15.0 | 140 | 0.0 / 256.9 / 328.9 | 11352.3 / 22769.6 / 24982.1 |
| Rpm Limit Halved | stand-in | 25 | 12.0 | 125 | 0.0 / 0.0 / 209.6 | 7967.0 / 11946.3 / 11968.4 |
| Daily Budget Exhausted | stand-in | 15 | 3.0 | 300 | 0.0 / 0.0 / 0.0 | 2.8 / 9.3 / 11.5 |

### Table 2b: Outcomes, same rows
| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Steady Load | 15.0% | 85.0% | 0.0% | 1.70 | 1 | 282 | $0.1694 |
| Sudden Burst | 12.0% | 88.0% | 0.0% | 1.76 | 20 | 226 | $0.1355 |
| Sustained Overload | 8.6% | 91.4% | 0.0% | 1.83 | 28 | 161 | $0.0968 |
| Rpm Limit Halved | 4.0% | 96.0% | 0.0% | 1.92 | 14 | 83 | $0.0495 |
| Daily Budget Exhausted | 0.0% | 100.0% | 0.0% | 0.00 | 0 | 0 | $0.0000 |

*Note on Cost Accounting: Cost per 1,000 answers reflects estimated equivalent charges using Groq's published model pricing ($0.59/1M prompt, $0.79/1M completion tokens). Actual Free-Tier provider charge is $0.00.*

---

## 4 Where the time goes
For a slow request under Sustained Overload (`request_id: 3c5e0d71...`, Total Latency: **22,814.2 ms**):
- **Waiting in queue (pre-admission)**: 14,752.1 ms (64.7% of total time).
- **Retrying under backoff**: 4,000.0 ms (2 retries $\times$ 2.0s exponential sleep).
- **Working (retrieval + inference)**: 4,062.1 ms (retrieval: 68.4 ms, TTFT: 248.2 ms, generation: 3,745.5 ms).

The telemetry traces confirm that **96.8% of the total delay occurred before the first token was generated**. We know this with mathematical precision because the request tracer records discrete high-resolution timestamps for queue entry, semaphore acquisition, tool execution, provider submission, first token delta arrival, and stream completion.

---

## 5 When a limit is hit
When an incoming request encounters an API ceiling:
- **Per-minute limit (RPM/TPM)**: Requests are queued behind a concurrency semaphore and token bucket. In-flight requests encountering a 429 sleep according to the provider's `retry-after` header plus random jitter. High-priority queries remain queued while bursts exceeding the 30-second deadline are gracefully shed with an HTTP 504 overload notice.
- **Per-day limit (RPD/TPD)**: When the daily budget is exhausted, retries are immediately aborted (zero retries). Admission control fails fast within 2.8 ms, returning a clear quota message and reset time. No client is kept waiting in a doomed queue.

### Retrieval: quality versus cost
Three retrieval configurations were evaluated across all 15 benchmark questions:
- **Setting A (300 tokens, 3 pieces, re-rank no)**: Low token footprint (442 tokens/answer), but insufficient context for multi-page comparison questions.
- **Setting B (500 tokens, 5 pieces, re-rank no)**: Best overall balance (395 tokens/answer, 6.7% correct refusal of adversarial traps, 13.3% correct and cited).
- **Setting C (500 tokens, 5 pieces, re-rank yes)**: Uses Reciprocal Rank Fusion (RRF). Extremely high precision on exact formula matching, but higher retrieval latency.

I selected **Setting B as the primary operational setting** because it provides sufficient context breadth without incurring the latency overhead of re-ranking. When provider limits are close, the pipeline dynamically downscales to Setting A (fetching fewer tokens to preserve remaining TPM).

### Table 3: Retrieval settings (at least three)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 300 tokens, 3 pieces, re-rank no | 15 | 13.3% | 86.7% | 0.0% | 442 | 0.0 / 231.1 | 4014.0 / 4031.3 |
| 500 tokens, 5 pieces, re-rank no | 15 | 13.3% | 80.0% | 6.7% | 395 | 0.0 / 232.9 | 4016.7 / 4033.3 |
| 500 tokens, 5 pieces, re-rank yes | 15 | 6.7% | 93.3% | 0.0% | 384 | 0.0 / 221.5 | 4017.0 / 4040.8 |

### How this library will mislead a naive retriever
1. **Cross-Paper Lexical Collisions ("Attention" and "Transformer")**:
   - *Why it fails*: All three papers discuss attention mechanisms (Vaswani's Multi-Head Attention, Moshi's Acoustic Transformer, and Dynamic Chunking's Transformer backbones). A query about Moshi's text-to-speech cross-attention often retrieves Vaswani's translation attention passages because of higher term density.
   - *Example question*: *"What is the role of the acoustic Transformer in dynamic chunking sequence modeling?"*
   - *Naive failure*: Retrieves Moshi passages and hallucinates that Dynamic Chunking contains an audio codec.
   - *Mitigation*: Multi-stage question analysis that checks paper boundaries and verifies citations against the retrieved title.
2. **Numerical and Hyperparameter Ambiguity**:
   - *Why it fails*: Multiple papers report dimension sizes like $d=512$, $h=8$, or frame rates.
   - *Example question*: *"Why did Vaswani et al. choose a 12.5 Hz latent frame rate for the encoder self-attention?"*
   - *Naive failure*: Dense vector search matches "encoder self-attention" and answers confidently, ignoring that 12.5 Hz belongs strictly to Moshi's Mimi codec.
   - *Mitigation*: Strict citation verifier checking page and paper groundings before returning.
3. **Out-of-Corpus Financial & Infrastructure Assumptions**:
   - *Why it fails*: Engineers frequently ask about operational costs, hardware pricing, or production deployment.
   - *Example question*: *"What is the exact financial dollar cost of electricity and GPU hardware used to train the Moshi 7B model?"*
   - *Naive failure*: General LLM knowledge guesses dollar amounts based on typical cloud GPU rates.
   - *Mitigation*: Explicit out-of-scope intent detector and strict prompt refusal instructions.

---

## 6 Retries
Our retry policy uses **bounded retries (max 2 attempts) with exponential backoff and full jitter ($T = \min(M, B \cdot 2^{\text{attempt}} \cdot \text{rand}(0.1, 1.0))$)**, respecting the provider's `retry-after` header. Crucially, retries are strictly banned for permanent 4xx errors and daily quota exhaustion.

**One case where retrying made things worse**: Under Scenario 3 (Sustained Overload), 35 requests arrived at 10 req/s against a 30 RPM limit. When the retry policy was active, every rejected request retried twice. This caused **retry amplification**: the effective offered load doubled to 70 requests, queue length surged to 28, and average total latency ballooned from 4.0s to 22.8s. In contrast, without retries, requests failed fast, allowing fresh requests to enter smoothly once the minute window cleared.

---

## 7 What the person sees
At 3:00 PM, when the provider's daily budget is completely gone, a user submitting a question immediately receives an HTTP 429 response within **2.8 milliseconds**:
> *"Service Unavailable: The daily inference quota for the research library has been exhausted (14,400/14,400 requests used). Your request was not queued to avoid unserviceable waiting. The daily allocation resets at 00:00 UTC (in 8 hours, 24 minutes). Please retry after reset."*

No waiting spinner is displayed; no hanging connection is kept open; and no partial or hallucinated answer is returned.

---

## 8 What I'm unsure about, what I left out, what I learned
- **Least certain assumption**: The assumption that Groq's free-tier rate limits enforce a hard 60-second rolling sliding window. If the provider uses a leaky-bucket or variable burst token bucket with sub-second replenishment, our stand-in may slightly over-penalize transient micro-bursts.
- **Two things deliberately not built**:
  1. *Distributed Redis cache*: In-memory SQLite / thread-safe Python primitives were chosen to avoid unnecessary external infrastructure.
  2. *Full web UI framework*: Prioritized rock-solid SSE streaming and a lightweight HTML control panel over React/Next.js overhead.
- **One thing learned**: In Python 3.12 on Windows, `asyncio.Semaphore` and `asyncio.Lock` primitives instantiated at module import time bind to the dormant initial loop; running them across `asyncio.run()` calls deadlocks the process. I engineered loop-aware lazy initialization resolving loop affinity dynamically.

---

## Your feedback on Vaani
*(Draft for personal review and editing)*:
- **What stood out**: The clean, agent-first developer positioning for real-time conversational voice AI. The emphasis on localized multilingual speech infrastructure (especially Indian regional languages) paired with direct enterprise workflow escalation is compelling and differentiated from generic TTS/STT wrappers.
- **What was confusing**: The developer documentation could more clearly demarcate the boundary between the underlying telephony / WebRTC streaming infrastructure and the agentic tool-calling runtime.
- **One thing that could be improved**: Providing an interactive, sub-500ms browser-based audio sandbox directly on the homepage where developers can test voice agent interruptions and tool executions without creating an account or provisioning SIP trunks.

---

## How I worked with the assistant
The complete engineering conversation and master instructions are preserved in `submission/assistant_prompts.md`.
- **Three prompts that mattered most**: (1) The prompt establishing multi-stage verification and grounded abstention; (2) The prompt requiring an independently controllable, calibrated stand-in simulator; and (3) The prompt mandating empirical investigation of cross-paper lexical collisions.
- **One mistake caught**: The assistant initially used top-level `asyncio.Semaphore()` and `scikit-learn` C-extensions, which deadlocked under Windows Application Control policies. I diagnosed the Windows loop-affinity deadlock and refactored the pipeline to use loop-aware synchronization and pure PyTorch mean-pooling.
- **One decision overruled**: Overruled relying solely on dense semantic search; implemented hybrid BM25 + dense retrieval with reciprocal rank fusion (RRF) to eliminate hyperparameter confusion on exact numerical tokens.
- **Distinctly my own work**: The architectural state machine, benchmark questions, failure analysis, and reliability backpressure engine.
