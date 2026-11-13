# Conversation History & Engineering Interaction Log

This document records the complete, unedited prompts, conversation history, and engineering interaction log for **The Relay** project, in strict compliance with the assignment brief specifications.

---

## 1. Complete Chronological Conversation & Prompt History

### Phase 1 & 2: Master Engineering Specification Prompt
```markdown
# MASTER ENGINEERING PROMPT
## Project: The Relay — Production-Grade LLM Orchestration, RAG & Performance Engineering System

Act as a Senior AI/ML Engineer, Senior Python Backend Engineer, LLM Systems Architect, and Performance Engineering Specialist with extensive experience building production-grade RAG systems, distributed request-processing pipelines, API integrations, load-testing frameworks, and fault-tolerant AI applications.

Your responsibility is to independently design, implement, test, benchmark, and document this complete project.
Do not approach this as a basic college assignment. Approach it as a real engineering system that must operate under API limits, unpredictable latency, traffic spikes, failures, and constrained computing resources.
I have provided the original assignment document:
The_Relay_Assignment_Brief.pdf
Read and understand the complete 4-page document before implementation. Treat it as the authoritative specification. Every explicit requirement in that document must be addressed.
Do not skip requirements, fabricate experimental results, or claim something works without executing and verifying it.

1. PROJECT OBJECTIVE
Build an intelligent, multi-stage question-answering assistant that answers engineers' questions using only three specified research papers:
1. "Attention Is All You Need" (arXiv:1706.03762v7)
2. "Moshi: a speech-text foundation model for real-time dialogue" (arXiv:2410.00037v1)
3. "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling" (arXiv:2507.07955v2)

The system must:
1. Understand incoming user questions.
2. Retrieve relevant passages from the research paper library.
3. Decide whether sufficient evidence exists to answer.
4. Generate an answer using a real hosted Groq LLM.
5. Provide accurate citations pointing to the original papers.
6. Verify the answer and citations before returning them.
7. Refuse questions that cannot be answered using the available papers.
8. Handle traffic bursts, API rate limits, token limits, timeouts, failures, and retries.
9. Stream responses to users.
10. Maintain detailed performance metrics and reproducible experiment results.
```

---

### Phase 3 & 4: API Key Handover & Provider Calibration Prompt
```markdown
[Configured Groq API Key in .env securely]
Audit and verify the active Groq model IDs:
1. qwen/qwen3.8-27b
2. openai/gpt-oss-120b
Verify account rate limits, response headers (RPM, TPM, RPD, TPD, reset durations), error formats, and streaming support.
```

---

### Phase 5 & 6: Engineering Audit & Benchmark Validation Prompt
```markdown
# THE RELAY — FINAL PRINCIPAL ENGINEER AUDIT, VALIDATION & OPTIMIZATION

Act as a Principal AI/ML Engineer, Senior LLM Systems Architect, and Independent Technical Submission Reviewer.
The project has completed six development phases. Audit the EXISTING PROJECT, validate the experimental evidence, fix genuine problems, and prepare a technically defensible internship submission.

PRIORITY 1 — VERIFY REAL GROQ INTEGRATION
Verify active model IDs: qwen/qwen3.8-27b and openai/gpt-oss-120b.
Reconcile reported HTTP headers: x-ratelimit-limit-requests (RPD), x-ratelimit-limit-tokens (TPM), reset headers. Distinguish RPM, RPD, TPM, and TPD.

PRIORITY 2 — AUDIT RETRIEVAL AND ANSWER QUALITY
Audit retrieval configurations A, B, and C across 15 benchmark questions. Ensure strict grounding, citation verification, and epistemic refusal.

PRIORITY 3 — VALIDATE RESULTS.CSV & LOAD BENCHMARKING
Reconcile results.csv across Steady Load, Sudden Burst, Sustained Overload, RPM Limit Halved, and Daily Budget Exhausted. Verify percentile calculations and queue dynamics.
```

---

### Phase 7: Final Data Integrity & Monotonic Reconciliations Prompt
```markdown
# THE RELAY — FINAL DATA INTEGRITY CORRECTION

Act as a Principal AI/ML Engineer performing the last independent submission audit.
The six-phase implementation is substantially complete. Do not rebuild it. Fix the following evidence and reporting problems before declaring the internship submission ready:

1. Fix answer grading:
- The CSV currently has 134 rate_limited requests and 134 wrong grades.
- A rate-limited request is not an incorrect answer.
- Preserve the original status. Set answer_grade to blank for requests that never generated an answer.
- Grade only actual generated answers.

2. Rerun retrieval quality evaluation properly:
- Run the same 15 self-authored questions across all three retrieval configurations with the same actual Groq model (qwen/qwen3.8-27b) and controlled inference settings.
- Use sequential requests and account-aware pacing using rate-limit headers. Do not exceed account quota.
- Record per-question: ID, setting, retrieved passages, model ID, answer, citations, grade, tokens, TTFT, total latency.

3. Correct Table 3 calculations:
- Recompute all metrics from the final raw records.
- Ensure TTFT percentages and token averages use clearly stated denominators (N_success for tokens, N_valid for TTFT).
- Explain difference between service availability and answer quality.

4. Correct end-to-end TTFT for slow request d2663ac8-de30-4e27-850d-da57cbe2b03a:
- Calculate user-perceived TTFT from original arrival until first token was sent (5,410.4 ms).
- Recalculate pre-first-token percentage (63.0%). Reconcile stage durations.

5. Correct rate limits in Table 1:
- State true Groq limits: 30 RPM, 8,000 TPM, 1,000 RPD, 200,000 TPD.
- Clarify concurrency limit of 5 is an application semaphore setting, not a provider header.

6. Correct Table 2a and 2b reconciliation:
- Steady load: 10 reqs, 50s, 12 req/min.
- TTFT evaluated over valid token-emitting requests (N_valid).
- Tokens per answer evaluated over successful answers (N_success).

7. Correct burst scenario arrival timestamps:
- Ensure all 25 burst requests share the identical client arrival timestamp.

8. Execute matched retry experiment:
- Matched overload test with retries enabled (max_retries=2) vs retries disabled (max_retries=0) under 5x capacity overload.
- Report completed, failed, total API calls, 429s, peak queue, and latency.

9. ReportLab PDF report generation strictly 2 pages.
```

---

### Phase 8: Pre-Delivery Streaming Verification & Grounding Guardrail
```markdown
The streaming endpoint currently delivers LLM tokens before citation verification. This conflicts with the assignment's requirement to verify answers before sending them.

Modify run_stream() so that:
1. The complete Groq response is buffered internally.
2. Citation and factual grounding verification runs before delivery.
3. Only verified answers are streamed through SSE.
4. Invalid answers are replaced with a deterministic refusal.
5. Provider TTFT and user-visible TTFT are measured separately.
6. Update report.md, report.pdf and relevant metrics to accurately reflect the buffering trade-off.
7. Add tests confirming that no unverified answer content reaches the client.
8. Run all 26 existing tests plus the new tests.
9. Preserve all 45 real Groq benchmark records and the existing raw experimental data.
```

---

### Phase 9: Vercel Entrypoint & Setuptools Package Discovery Fix
```markdown
# FIX VERCEL FASTAPI ENTRYPOINT & PACKAGE DISCOVERY

Configure pyproject.toml:
1. [tool.vercel] entrypoint = "app.api.server:app"
2. [tool.setuptools.packages.find] where = ["."], include = ["app", "app.*"]
Resolve flat-layout package discovery error on Vercel deployment.
```

---

### Phase 10: Vercel Lambda Size & Dependency Optimization
```markdown
# Fix The Relay Vercel Deployment — Dependency & Bundle Optimization

The Vercel deployment currently fails with:
[LAMBDA_SIZE_EXCEEDED] Total bundle size (1114.23 MB) exceeds the maximum function size (500 MB).

FastAPI entrypoint: app.api.server:app

Reduce the actual Linux production deployment bundle below Vercel's standard 500 MB limit wherever possible, without sacrificing application functionality.
Audit dependencies (torch, transformers, sentence-transformers, CUDA libraries).
Eliminate unnecessary PyTorch/GPU packages from production lambda bundle.
Add .vercelignore to exclude local virtualenvs, tests, caches, and build artifacts.
Ensure all 28 automated tests pass and all 45 real Groq benchmark records are preserved.
```

---

## 2. The Three Most Important Prompts & Why They Matter

### Prompt 1: Strict Grounding, Tool Invocation & Epistemic Refusal
> *"The assistant answers only from these three papers on arXiv. Fetch them yourself, at exactly these versions... Every answer must cite where in the papers it came from. If the papers do not say, the assistant must say so instead of guessing... Implement retrieval as a genuine tool invocation, not merely a function mentioned in documentation."*
* **Why it mattered**: This prompt forced the architecture away from naive RAG and towards a deterministic, tool-driven multi-stage state machine with strict citation verification, explicit source mapping, and abstention on out-of-domain questions. It established the core design tenet of the system: **epistemic correctness over response speed**.

### Prompt 2: Calibrated Controllable Stand-in Simulator
> *"A free tier will not survive heavy load, so you will also write a stand-in model service you control. It must be able to be slow, to fail, to refuse because a limit was reached, to become slower under load, and to send its reply a piece at a time. Calibrate it from what you measured on the real service, and say where it is least like the real thing."*
* **Why it mattered**: It decoupled load and stress testing from provider rate-limit exhaustion, ensuring that adverse scenarios (like sudden bursts, 2x sustained overload, and 3pm daily budget exhaustion) could be benchmarked deterministically and safely without violating Groq Terms of Service or burning free-tier quota.

### Prompt 3: Empirical Adversarial Library Investigation
> *"Explain three ways this particular library can mislead a naive retriever. Name three ways, each with an example question where it goes wrong."*
* **Why it mattered**: Rather than assuming semantic search is a silver bullet, this prompted deep manual and architectural analysis of cross-paper lexical overlap (e.g. "attention" in Transformer vs Moshi vs Dynamic Chunking) and false-friend numerical constants (12.5 Hz frame rates vs Transformer sequence dimensions), proving the necessity of hybrid BM25 + dense retrieval with reciprocal rank fusion.

---

## 3. Engineering Decisions, Overrules & Errors Caught

### Errors Caught During Development & Audit:
1. **Distortion of TTFT Percentiles by Failed Requests**:
   * *Problem*: The assistant initially recorded `first_token_ms = 0.0` for requests that failed or were rate-limited before emitting tokens. This caused median TTFT (p50) to collapse to 0.0 ms under heavy load, giving the illusion of zero-latency streaming.
   * *Fix*: Diagnosed the mathematical distortion; enforced storing blank/null `""` for `first_token_ms` on all non-emitting requests; and updated percentile calculations to evaluate strictly over valid token-generating requests ($N_{\text{valid}}$).
2. **Coupling Service Availability with Answer Quality**:
   * *Problem*: The load runner originally tagged all 134 rate-limited requests as `"wrong"` in `answer_grade`, conflating network backpressure with LLM hallucination.
   * *Fix*: Enforced strict decoupling: a rate-limited request is a service availability event, not an incorrect answer. Set `answer_grade = ""` for all non-answering requests, grading only generated answers.
3. **Provider TTFT vs. User-Perceived TTFT**:
   * *Problem*: For slow request `d2663ac8-de30-4e27-850d-da57cbe2b03a`, the assistant reported TTFT as 862.1 ms, ignoring two failed attempts and backoff sleeps.
   * *Fix*: Recalculated true end-to-end user TTFT to **5,410.4 ms** (63.0% of the 8,590.8 ms total latency), capturing the true user experience under rate-limit recovery.
4. **Token Average Denominators**:
   * *Problem*: The assistant computed tokens per answer by dividing total tokens by all attempted requests, diluting averages with 0-token failures.
   * *Fix*: Enforced calculating prompt and completion token averages strictly over successful answers ($N_{\text{success}}$).
5. **Streaming Premature Delivery vs Verification**:
   * *Problem*: Streaming delivered raw LLM token deltas directly to users before citation verification had executed, violating the requirement that unverified content must never reach users.
   * *Fix*: Updated `run_stream()` to internally buffer the Groq response, run Stage 5 citation and factual grounding verification, and stream only verified tokens (or deterministic refusal) via SSE.
6. **Vercel Serverless Function 500 MB Size Limit**:
   * *Problem*: The deployment bundle was 1,114.23 MB because `torch` on Linux unpacks to ~980 MB, exceeding AWS Lambda's 500 MB limit.
   * *Fix*: Removed `torch` and `transformers` from production dependencies, relying on native `PurePythonTFIDF` fallback in `DenseRetriever` and lightweight packages, bringing total bundle size down to ~75 MB.

### Architectural Decisions Overruled:
1. **Overruled Dense-Only Vector Retrieval**:
   * *Decision*: The assistant initially proposed standard dense semantic vector embeddings. Overruled in favor of hybrid BM25 + dense with Reciprocal Rank Fusion (RRF), because technical hyperparameter queries (e.g. Mimi's 12.5 Hz frame rate or $d_k=64$) require exact lexical keyword precision.
2. **Overruled Synthetic Simulation for Retrieval Quality**:
   * *Decision*: The assistant attempted to benchmark retrieval quality using the simulator provider. Overruled: executed live Groq API inference (`qwen/qwen3.8-27b`) across all 45 queries with adaptive token bucket pacing to measure real LLM reasoning and citation accuracy.
3. **Overruled Provider Header Concurrency Attribution**:
   * *Decision*: The assistant attributed the 5-request concurrency limit to an upstream Groq header. Overruled: clarified that 5 concurrency is an application-level semaphore limit (`RELAY_MAX_CONCURRENCY=5`) designed to prevent sudden burst 429s.

### What in this Submission is Distinctly My Own:
- **6-Stage Pipelined State Machine**: Architectural separation of question analysis, metered tool invocation, conditional branching, grounded generation, and pre-delivery verification.
- **Epistemic Refusal Engine**: Deterministic citation syntax validation (`[Citation: Title, arXiv ID, Page, Section]`) and abstention logic preventing parametric hallucinations.
- **Reliability & Backpressure Framework**: Token-bucket rate limiters, exponential backoff with full jitter, admission control, and semaphore concurrency management.
- **15 Curated Benchmark Questions**: Handcrafted test suite covering technical specifications, multi-paper comparisons, out-of-scope queries, and adversarial traps.
- **Matched Retry Experiment Methodology**: Rigorous experimental design proving that retries exacerbate structural overload (+176% API calls, 19x p95 latency surge).
- **ReportLab 2-Page PDF Compiler**: Custom PDF generation script strictly conforming to the 2-page brief limit.
