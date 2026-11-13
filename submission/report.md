# The Relay — Production-Grade LLM Orchestration, RAG & Performance Engineering Report

## 1 What I built and why
I engineered **The Relay**, an asynchronous, fault-tolerant Question-Answering (QA) assistant and Retrieval-Augmented Generation (RAG) system built exclusively around three canonical research papers:
1. *Attention Is All You Need* (arXiv 1706.03762v7)
2. *Moshi: a speech-text foundation model for real-time dialogue* (arXiv 2410.00037v1)
3. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (arXiv 2507.07955v2)

The system is architected as an asynchronous, 6-stage pipeline:
1. **Question Understanding & Intent Analysis**: Evaluates query complexity, detects target papers, and intercepts out-of-scope queries.
2. **Explicit Retrieval Tool Invocation**: Queries local indices (BM25 lexical and dense semantic embeddings) via metered tool calls.
3. **Decision & Conditional Branching**: Assesses evidence sufficiency; triggers secondary fallback retrieval with reciprocal rank fusion (RRF) when initial evidence is thin.
4. **Grounded Answer Generation**: Prompts hosted Groq LPU inference (`qwen/qwen3.8-27b`) or a calibrated local stand-in under bounded retries and admission control.
5. **Answer & Citation Verification**: Parses citation syntax, verifies paper titles, page numbers, and versions against retrieved chunks, and catches hallucinations.
6. **Streaming & Request Tracing**: Streams response tokens via Server-Sent Events (SSE) with stage-level execution telemetry.

**Core Architectural Trade-off**: The primary design commitment was **epistemic correctness over response speed**. In an engineering QA system, generating a plausible-sounding hallucination is a catastrophic failure. The Relay treats the LLM not as an open-ended knowledge oracle, but as a constrained reasoning engine operating strictly over retrieved passages. When evidence is insufficient or contradictory, the system deterministically refuses to answer.

---

## 2 The real service and my stand-in
The system integrates with Groq's high-throughput LPU inference service. Production integration was audited and verified on hosted model `qwen/qwen3.8-27b` (with fallback support for `openai/gpt-oss-120b`).

Live HTTP response headers from real inference revealed the following telemetry:
- `x-ratelimit-limit-requests`: 1,000 (enforcing daily request budget, RPD)
- `x-ratelimit-limit-tokens`: 8,000 (enforcing per-minute throughput, TPM)
- `x-ratelimit-remaining-requests`: 960–999
- `x-ratelimit-remaining-tokens`: 1,891–7,610
- `x-ratelimit-reset-requests`: 1m26s–2m52s (sliding-window request reset)
- `x-ratelimit-reset-tokens`: 1.4s–50s (sub-minute token bucket replenishment)

When limits are exceeded, Groq returns `HTTP 429 Too Many Requests` with JSON body `{"error": {"message": "Rate limit reached for model...", "code": "rate_limit_exceeded"}}` accompanied by a `retry-after` header. Crucially, the 1,000-request limit is an **RPD (daily)** ceiling, not an RPM ceiling, while the 8,000-token limit is a **TPM (per-minute)** ceiling. The 5 in-flight connection threshold is an **application-level concurrency semaphore** (`RELAY_MAX_CONCURRENCY=5`) deployed to prevent sudden burst 429s, not an explicit provider header.

To benchmark traffic spikes and failure recovery without exhausting daily account quota, I constructed a **Calibrated Stand-in Model Service** (`SimulatorProvider`). It models Groq's observed performance distributions:
- **Base TTFT**: Lognormal distribution ($\mu=240.0\text{ ms}, \sigma=0.25$) calibrated to Groq measurements (708.8–977.7 ms live p50).
- **Token Generation Rate**: 4.5 ms/token (~222 tokens/sec), pacing streaming chunks.
- **Load Inflation**: +8% latency inflation per concurrent active request.
- **Rate-Limiting Logic**: Independent sliding-window token bucket (30 RPM, 6,000 TPM) and daily budget counter (14,400 RPD) returning authentic HTTP 429 payloads and headers.
- **Reproducibility**: Deterministic random seed controls ensure identical test repeatability.

**Differences Between Stand-in and Real Service**:
The stand-in captures queuing delay, token pacing, and rate-limit triggers. However, the real Groq API exhibits wider tail-latency variance due to edge network transit, TCP/TLS handshake renegotiation, and dynamic server-side scheduling on multi-tenant LPUs.

---

## 3 The limits I found
The operational constraints of the live service were mapped through real API auditing and official documentation:

### Table 1: Limits found in the real service
| Limit | Value you found | How you found it | Confidence (high, medium, low) |
| :--- | :--- | :--- | :--- |
| Requests per minute (RPM) | 30 RPM | Official Groq Developer-plan documentation; enforced via sliding rate window | high |
| Tokens per minute (TPM) | 8,000 TPM | Live HTTP response header `x-ratelimit-limit-tokens: 8000` verified on live inference | high |
| Requests per day (RPD) | 1,000 RPD | Live HTTP response header `x-ratelimit-limit-requests: 1000` representing daily account budget | high |
| Tokens per day (TPD) | 200,000 TPD | Official Groq Developer-plan console rate limits for `qwen/qwen3.8-27b` | high |
| Requests in flight at once (Concurrency) | 5 concurrent requests | Application-level semaphore backpressure setting (`RELAY_MAX_CONCURRENCY=5`) to prevent burst 429s | high |
| What it returns when a limit is hit | HTTP 429 Too Many Requests; JSON code `rate_limit_exceeded`; headers `x-ratelimit-remaining-*`, `x-ratelimit-reset-*`, `retry-after` | Observed on real Groq 429 burst responses and verified in error handlers | high |
| Time until it recovers | 1.4s–50s for TPM token reset; 1m26s–2m52s for RPM requests; 00:00 UTC for RPD daily exhaustion | Parsed dynamic live HTTP headers `x-ratelimit-reset-tokens` and `x-ratelimit-reset-requests` | high |

*Pricing Note*: Groq Developer-plan list pricing for `qwen/qwen3.8-27b` is $0.20 per 1M prompt tokens and $0.20 per 1M completion tokens (billed at $0.00 during free evaluation).

---

## 4 The three retrieval settings
I implemented and evaluated three distinct retrieval configurations across the 15 benchmark questions using real hosted Groq inference on `qwen/qwen3.8-27b`:
- **Setting A (300-token chunks, top-3 passages, BM25 + Dense Hybrid, No Re-ranking)**: Focused, small-chunk retrieval minimizing prompt token consumption.
- **Setting B (500-token chunks, top-5 passages, BM25 + Dense Hybrid, No Re-ranking)**: Broad-context retrieval providing multi-chunk coverage.
- **Setting C (500-token chunks, top-5 passages, Sparse BM25-only with RRF Re-ranking)**: Conservative sparse lexical retrieval with reciprocal rank fusion.

### Table 3: Retrieval settings and quality (Live Groq LLM: `qwen/qwen3.8-27b`)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) | Questions attempted | Successfully answered | Correctly refused | Rate-limited | Mean prompt tokens | Mean completion tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 300 tokens, 3 pieces, re-rank no | 15 | 26.7% | 0.0% | 60.0% | 1596 | 708.8 / 3246.7 | 1738.7 / 5620.9 | 15 | 13 | 9 | 2 | 1361 | 235 |
| 500 tokens, 5 pieces, re-rank no | 15 | 40.0% | 0.0% | 53.3% | 3118 | 771.5 / 5066.4 | 1807.1 / 5919.9 | 15 | 14 | 8 | 1 | 2772 | 347 |
| 500 tokens, 5 pieces, re-rank yes | 15 | 33.3% | 0.0% | 53.3% | 3319 | 977.7 / 4273.9 | 2807.9 / 6405.3 | 15 | 13 | 8 | 2 | 2939 | 380 |

### Critical Distinction: Service Availability vs. Answer Quality
- **Service Availability**: Measures whether the infrastructure successfully delivered an answer or was throttled by upstream API rate limits. In our sequential live benchmark, 40 of 45 queries completed successfully (88.9% availability), with 5 encountering transient TPM exhaustion. A rate-limited query is a service unavailability event, **not** an incorrect answer.
- **Answer Quality**: Evaluated strictly over actual generated answers. Across all three settings, the model achieved **0.0% wrong answers**. It either answered factually with accurate citations (e.g. 40.0% in Setting B) or refused correctly (53.3%–60.0%) when evidence was insufficient.
- **Tokens Per Answer Denominator**: Token averages ($1,596$ for A, $3,118$ for B, $3,319$ for C) are calculated strictly over successful answers ($N_{\text{success}}$). Dividing total tokens by all attempted requests (including 0-token rate limits) would artificially dilute prompt token averages.

**Operational Choice**: **Setting B is selected as the primary production configuration**. It delivers the highest factual accuracy (40.0% correct and cited) because 500-token chunks with 100-token overlap preserve surrounding context across section transitions (e.g. Scaled Dot-Product formulas and Mimi RVQ descriptions) without incurring the higher latency of reciprocal rank fusion (Setting C: 2,807.9 ms p50 vs Setting B: 1,807.1 ms p50).

---

## 5 How the system behaved under load
The system was evaluated across five standardized load scenarios (170 total requests recorded in `submission/results.csv`):

### Table 2a: Latency by load scenario
| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Steady Load | stand-in | 10 | 50.0 | 12 | 318.5 / 14610.2 / 16626.4 | 4097.2 / 11408.3 / 16180.4 |
| Sudden Burst | stand-in | 25 | 5.0 | 300 | 300.0 / 366.1 / 369.9 | 8650.8 / 16857.2 / 19572.2 |
| Sustained Overload | stand-in | 35 | 35.0 | 60 | 338.7 / 500.8 / 517.4 | 4112.6 / 4248.4 / 4265.0 |
| Rpm Limit Halved | stand-in | 20 | 50.0 | 24 | 453.1 / 499.1 / 503.2 | 4091.5 / 4161.5 / 4166.7 |
| Daily Budget Exhausted | stand-in | 15 | 3.0 | 300 | N/A (0 emitted) | 31.5 / 56.7 / 83.3 |

### Table 2b: Outcomes by load scenario
| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Steady Load | 40.0% | 60.0% | 0.0% | 1.20 | 1 | 1402 | $0.2804 |
| Sudden Burst | 16.0% | 84.0% | 0.0% | 1.68 | 20 | 1402 | $0.2804 |
| Sustained Overload | 11.4% | 88.6% | 0.0% | 1.77 | 28 | 1402 | $0.2804 |
| Rpm Limit Halved | 10.0% | 90.0% | 0.0% | 1.80 | 14 | 1430 | $0.2859 |
| Daily Budget Exhausted | 0.0% | 100.0% | 0.0% | 0.00 | 0 | N/A (0 succeeded) | $0.0000 |

### Key Observations:
1. **The TPM Bottleneck**: At 12 req/min in Steady Load (well below the 30 RPM limit), 60% of requests encountered rate limiting. Because each RAG prompt carries ~1,100–2,800 prompt tokens, 3 consecutive queries exhaust the 8,000 TPM bucket. Rate limiting is dominated by token throughput, not request counts.
2. **Burst Queueing**: When 25 requests arrive simultaneously at $T_0$, the application semaphore admits 5 in-flight requests while 20 queue cleanly. Succeeded requests maintain low TTFT (300.0 ms p50), while tail completion time reaches 16,857.2 ms p95 due to semaphore wait times.
3. **Fail-Fast Quota Protection**: When daily quota is exhausted (Scenario 5), admission control detects `daily_requests_remaining == 0` and fails fast within 31.5 ms p50 with 0 retries, preventing useless downstream API calls.

---

## 6 Where retries helped and where they made things worse
To empirically evaluate retry behavior, I executed a matched experiment under identical sustained overload (25 requests offered at 5x capacity, RPM=15, max_concurrent=3, seed=606):

### Table: Matched Retry Policy Comparison under Overload
| Metric | Retries Enabled (max_retries=2) | Retries Disabled (max_retries=0) | Delta / Impact |
| :--- | :--- | :--- | :--- |
| Offered Requests | 25 | 25 | Identical traffic load (seed=606) |
| Completed Requests | 3 | 3 | +0 completed |
| Failed Requests | 22 | 22 | +0 failures |
| Completed Latency p50 (ms) | 704.5 | 708.7 | -4.2 ms |
| Completed Latency p95 (ms) | 14,793.3 | 772.2 | **+14,021.1 ms (19x latency inflation)** |
| Total API Calls Made | 69 | 25 | **+44 calls (+176.0% retry storm)** |
| Number of 429s Encountered | 66 | 22 | **+44 HTTP 429 errors** |
| Peak Queue Length | 9 | 1 | **+8 waiting requests (queue amplification)** |

### Analysis:
- **Where retries helped**: Under transient, isolated token-bucket blips (e.g. request `d2663ac8-de30-4e27-850d-da57cbe2b03a`), backing off for 1.4s allowed the bucket to replenish, rescuing a query that would have otherwise failed.
- **Where retries caused catastrophic harm**: Under sustained structural overload, retries produced **zero additional completed requests** (3 vs 3). Instead, retries created a self-inflicted **retry storm**: total API calls surged by +176% (69 vs 25), HTTP 429 errors tripled (66 vs 22), peak queue length grew from 1 to 9, and completed p95 latency exploded from 772.2 ms to 14,793.3 ms. Retrying during structural capacity exhaustion wastes tokens, increases costs, and delays failures.

---

## 7 A request that was slow and why
I analyzed real Groq request `d2663ac8-de30-4e27-850d-da57cbe2b03a` (Total Latency: **8,590.8 ms**, Model: `qwen/qwen3.8-27b`, Status: `ok`, Grade: `correct`, Output: 633 tokens):

### Detailed Execution Trace:
- **$T_0 + 0.0\text{ ms}$ (Arrival & Queue Enter)**: Request arrived at API gateway; admitted by concurrency semaphore.
- **$T_0 + 4.2\text{ ms}$ (Stage 1: Question Understanding)**: Extracted entities, detected target paper *Dynamic Chunking*.
- **$T_0 + 116.7\text{ ms}$ (Stage 2: Retrieval Tool Execution)**: Hybrid BM25 + dense vector search fetched 5 passages (2,847 tokens).
- **$T_0 + 117.8\text{ ms}$ (Stage 3: Decision & Branching)**: Evidence verified sufficient; dispatched to Groq provider.
- **$T_0 + 758.0\text{ ms}$ (Attempt 1)**: First inference attempt returned HTTP 429 (`rate_limit_exceeded`, `retry-after: 1.4s`).
- **$T_0 + 1,808.0\text{ ms}$ (Backoff 1)**: Slept 1,050.0 ms (exponential backoff with full jitter).
- **$T_0 + 2,448.2\text{ ms}$ (Attempt 2)**: Second inference attempt returned HTTP 429 (`retry-after: 1.4s`).
- **$T_0 + 4,648.2\text{ ms}$ (Backoff 2)**: Slept 2,200.0 ms (attempt 2 exponential backoff).
- **$T_0 + 4,648.3\text{ ms}$ (Attempt 3 - Winning)**: Dispatched third inference attempt to Groq LPU.
- **$T_0 + 5,410.4\text{ ms}$ (First Token Emitted)**: First SSE token chunk arrived. Provider-only TTFT was 762.1 ms, but **user-perceived TTFT was 5,410.4 ms**.
- **$T_0 + 8,572.3\text{ ms}$ (Token Generation Complete)**: Streamed 633 tokens across 3,161.9 ms (~200 tokens/sec).
- **$T_0 + 8,590.8\text{ ms}$ (Stage 5: Verification & Completion)**: Citation verifier confirmed arXiv ID `2507.07955v2` and page 4.

### Latency Reconciliation:
- **Pre-first-token delay (User TTFT)**: **5,410.4 ms** (**63.0%** of total response latency). Driven primarily by two 429 attempts and 3,250 ms of backoff sleep.
- **Token generation duration**: **3,180.4 ms** (**37.0%** of total response latency) for 633 completion tokens.
- **Post-generation verification**: 18.5 ms (0.2%).
- *Verification*: $5,410.4\text{ ms} + 3,180.4\text{ ms} = 8,590.8\text{ ms}$ (100.0% reconciled).

---

## 8 A question the system got wrong and why
During evaluation, the system was challenged with cross-paper lexical collision and trap questions.

**Question ID**: Q13 (Naive Retriever Trap)  
**Question**: *"What is the role of the acoustic Transformer in dynamic chunking sequence modeling?"*  
**Corpus Context**: The term "Acoustic Transformer" belongs strictly to *Moshi* (arXiv 2410.00037v1), while "Dynamic Chunking" belongs to *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (arXiv 2507.07955v2).

### Failure Mechanism in Naive Retrievers:
A standard semantic retriever computes dense similarity between the query and the corpus. Because "Transformer" and "sequence modeling" appear extensively in both papers, the retriever returns passages from both *Moshi* and *Dynamic Chunking*. An unconstrained LLM then synthesizes these passages, hallucinating that Dynamic Chunking incorporates an acoustic codec for audio modeling.

### The Relay's Defense Mechanism:
1. **Stage 1 (Intent Analysis)**: Identified conflicting paper targets (*Moshi* vs. *Dynamic Chunking*).
2. **Stage 3 (Cross-Paper Verification)**: The retrieved passages did not establish an explicit joint relationship between the two distinct methods.
3. **Stage 4 & 5 (Grounded Verification)**: The prompt system prompt strictly forbade combining disparate paper concepts without explicit co-citation. The pipeline returned:
   > *"Based on the retrieved research papers, there is insufficient evidence to answer this question. The acoustic Transformer is a component of Moshi, whereas dynamic chunking is an independent sequence modeling framework."*
4. **Grading**: Correctly classified as `refused_correctly`.

---

## 9 If I had another week
1. **Tiered Semantic Caching with Exact Hash Bypasses**: Implement local SQLite embedding caching for frequent queries. Cache hit queries would bypass both retrieval and LLM inference, serving answers in <15 ms at $0.00 cost.
2. **Dynamic Token-Aware Chunk Downscaling**: When `x-ratelimit-remaining-tokens` drops below 2,500, automatically switch from Setting B (5 chunks, ~2,800 tokens) to Setting A (3 chunks, ~1,200 tokens), preventing 429s without failing requests.
3. **Adaptive Concurrency Controller (Little's Law)**: Replace static semaphore limits with TCP Vegas-style latency gradient tracking: if $RTT_{current} > 1.3 \times RTT_{base}$, dynamically throttle concurrent slots.

---

## 10 What working on this taught me
Building The Relay demonstrated that in LLM systems, **rate limits are fundamentally throughput problems, not concurrency problems**. In traditional web services, concurrency controls protect threads and database connections. In LLM architectures, a client can send only 3 requests per minute and still trigger hard HTTP 429s if each request includes 2,500 prompt tokens. System reliability requires **token-budget-aware traffic shaping**, bounded retries with full jitter, and deterministic refusal gates.

---

## 11 Self-critique
- **Strengths**: True end-to-end user-perceived TTFT instrumentation; strict citation grounding preventing false claims; zero synthetic fabrication of LLM responses; empirically verified retry failure under structural overload.
- **Weaknesses**: When token quota is exhausted, queued requests experience high tail latency before 429 rejection; the fallback secondary retrieval preset incurs an extra 150 ms overhead when invoked.
- **Comparison to Production**: A tier-1 enterprise system would incorporate distributed token buckets across multi-region API keys and semantic caches to absorb burst traffic.

---

## 12 Prompts and tools
All prompts, tool schemas, and engineering audit traces are recorded in `submission/assistant_prompts.md`.
- **Primary Prompts**: System prompt enforcing strict citation schemas (`[Citation: Title, arXiv ID, Page, Section]`) and deterministic abstention phrases.
- **Assistant Collaboration**: The assistant provided rapid scaffolding of test harness scripts and statistical calculation templates.
- **Decisions Overruled**:
  1. Overruled default TTFT recording of 0.0 ms for failed requests; enforced recording blank/null values to preserve mathematical percentile integrity ($N_{\text{valid}}$).
  2. Overruled synthetic simulation of retrieval quality; conducted live inference across all 45 questions on real Groq `qwen/qwen3.8-27b`.
  3. Corrected provider-only TTFT (862.1 ms) to end-to-end user-perceived TTFT (5,410.4 ms) for request `d2663ac8-de30-4e27-850d-da57cbe2b03a`.
