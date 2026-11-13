"""
Multi-Stage Pipeline Orchestrator for The Relay.
Executes the full 6-stage AI workflow:
1. Question Understanding
2. Retrieval Tool Execution
3. Decision & Conditional Branching / Secondary Retrieval Loop
4. Grounded Answer Generation (Groq / Stand-in)
5. Answer & Citation Verification
6. Streaming / Non-Streaming Delivery with Request Tracing
"""

import time
import uuid
import asyncio
import re
from typing import AsyncGenerator, Dict, Any, Optional, List

from app.schemas.models import (
    AskRequest,
    PipelineResult,
    PipelineTraceStep,
    RetrievedPassage,
    Citation,
    QuestionUnderstanding
)
from app.pipeline.question_analyzer import analyze_question
from app.tools.retrieval_tool import RetrievalTool, RetrievalToolInput
from app.pipeline.verifier import verify_answer, is_refusal
from app.providers.base import LLMProvider, ProviderResponse
from app.providers.groq_provider import GroqProvider
from app.providers.simulator_provider import SimulatorProvider
from app.core.config import settings
from app.core.reliability import reliability_mgr, compute_backoff

SYSTEM_PROMPT = """You are an authoritative research assistant answering engineers' questions based EXCLUSIVELY on the provided research papers.
The corpus contains only three papers:
1. "Attention Is All You Need" (arXiv 1706.03762v7)
2. "Moshi: a speech-text foundation model for real-time dialogue" (arXiv 2410.00037v1)
3. "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling" (arXiv 2507.07955v2)

STRICT RULES:
1. Base your answer ONLY on the provided retrieved passages.
2. If the passages do not contain enough information to answer fully, state clearly: "Based on the retrieved research papers, there is insufficient evidence to answer this question." Do NOT guess or use external general knowledge.
3. Every substantive statement MUST be followed by an exact citation in this format:
   [Citation: <Paper Title>, arXiv <arxiv_id><version>, Page <page_number>, Section <section>]
   Example: [Citation: Attention Is All You Need, arXiv 1706.03762v7, Page 3, Section 3.2]
"""

class PipelineOrchestrator:
    def __init__(self, provider: Optional[LLMProvider] = None, max_retries: int = 2):
        if provider:
            self.provider = provider
        elif settings.provider_mode == "real" and settings.groq_api_key:
            self.provider = GroqProvider()
        else:
            self.provider = SimulatorProvider()
        self.retrieval_tool = RetrievalTool()
        self.max_retries = max_retries

    def set_provider(self, provider: LLMProvider):
        self.provider = provider

    async def execute(
        self,
        request: AskRequest,
        request_id: Optional[str] = None
    ) -> PipelineResult:
        req_id = request_id or str(uuid.uuid4())
        t_global_start = time.perf_counter()
        traces: List[PipelineTraceStep] = []
        retries_count = 0

        # Concurrency & Queue wait tracking
        t_queue_start = time.perf_counter()
        await reliability_mgr.track_queue_enter()
        await reliability_mgr.get_semaphore().acquire()
        queue_wait_ms = (time.perf_counter() - t_queue_start) * 1000.0
        await reliability_mgr.track_queue_leave()

        try:
            # -------------------------------------------------------------
            # Stage 1: Question Understanding
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            understanding: QuestionUnderstanding = analyze_question(request.question)
            d1 = (time.perf_counter() - t0) * 1000.0
            traces.append(PipelineTraceStep(
                step_name="1_question_understanding",
                started_at_ms=(t0 - t_global_start) * 1000.0,
                duration_ms=d1,
                status="ok",
                metadata={"complexity": understanding.complexity, "terms": understanding.technical_terms}
            ))

            # Early refusal for blatant out-of-scope queries
            if not understanding.needs_retrieval:
                total_ms = (time.perf_counter() - t_global_start) * 1000.0
                await reliability_mgr.record_outcome("ok", retries=0)
                return PipelineResult(
                    request_id=req_id,
                    question=request.question,
                    answer="Based on the research library, this question cannot be answered. The provided corpus contains only three machine learning papers and does not contain information on this topic.",
                    refused=True,
                    refusal_reason="Out of corpus scope",
                    first_token_ms=total_ms,
                    total_ms=total_ms,
                    queue_wait_ms=queue_wait_ms,
                    status="ok",
                    traces=traces,
                    source=getattr(self.provider, "source", "stand-in"),
                    retrieval_setting=request.retrieval_setting,
                    answer_grade="refused_correctly"
                )

            # -------------------------------------------------------------
            # Stage 2: Tool Execution (Retrieval Tool)
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            tool_input = RetrievalToolInput(
                query=request.question,
                retrieval_setting=request.retrieval_setting,
                strategy="primary"
            )
            tool_output = await self.retrieval_tool.execute(tool_input)
            passages = tool_output.passages
            d2 = (time.perf_counter() - t0) * 1000.0
            traces.append(PipelineTraceStep(
                step_name="2_retrieval_tool",
                started_at_ms=(t0 - t_global_start) * 1000.0,
                duration_ms=d2,
                status="ok",
                metadata={"passages_count": len(passages), "tool_ms": tool_output.execution_time_ms}
            ))

            # -------------------------------------------------------------
            # Stage 3: Decision & Conditional Branching / Secondary Retrieval Loop
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            # If evidence is weak (no passages or top score very low), execute secondary retrieval
            needs_secondary = (len(passages) == 0) or (passages[0].score < 0.1 and request.retrieval_setting != "C")
            branch_action = "primary_sufficient"

            if needs_secondary:
                branch_action = "secondary_escalation"
                # Refine query with extracted technical terms
                refined_query = f"{request.question} {' '.join(understanding.technical_terms)}".strip()
                sec_input = RetrievalToolInput(
                    query=refined_query,
                    retrieval_setting="C",
                    strategy="secondary_fallback"
                )
                sec_output = await self.retrieval_tool.execute(sec_input)
                if sec_output.passages:
                    passages = sec_output.passages

            d3 = (time.perf_counter() - t0) * 1000.0
            traces.append(PipelineTraceStep(
                step_name="3_decision_and_branching",
                started_at_ms=(t0 - t_global_start) * 1000.0,
                duration_ms=d3,
                status="ok",
                metadata={"branch_action": branch_action, "final_passages": len(passages)}
            ))

            # If still zero passages found, refuse immediately
            if not passages:
                total_ms = (time.perf_counter() - t_global_start) * 1000.0
                await reliability_mgr.record_outcome("ok", retries=0)
                return PipelineResult(
                    request_id=req_id,
                    question=request.question,
                    answer="Based on the retrieved research papers, there is insufficient evidence to answer this question. The library does not contain relevant passages.",
                    refused=True,
                    refusal_reason="Zero relevant passages retrieved",
                    first_token_ms=total_ms,
                    total_ms=total_ms,
                    queue_wait_ms=queue_wait_ms,
                    status="ok",
                    traces=traces,
                    source=getattr(self.provider, "source", "stand-in"),
                    retrieval_setting=request.retrieval_setting,
                    answer_grade="refused_correctly"
                )

            # -------------------------------------------------------------
            # Stage 4: Answer Generation with Bounded Retries
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            context_blocks = []
            for p in passages:
                context_blocks.append(
                    f"--- Passage from '{p.paper_title}' (arXiv {p.arxiv_id}{p.version}), Page {p.page_number}, Section: {p.section} ---\n{p.text}"
                )
            context_str = "\n\n".join(context_blocks)
            user_prompt = f"Question: {request.question}\n\nRetrieved Passages from Corpus:\n{context_str}\n\nAnswer:"

            # Bounded retry loop (max_retries for transient rate limits, zero retries for daily budget)
            max_retries = self.max_retries
            resp: Optional[ProviderResponse] = None
            t_winning_attempt_start = t0

            for attempt in range(max_retries + 1):
                t_attempt_start = time.perf_counter()
                resp = await self.provider.generate(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
                if resp.status == "ok":
                    t_winning_attempt_start = t_attempt_start
                    break
                
                # Check permanent vs retryable failure
                is_daily = "daily" in (resp.error_message or "").lower() or (resp.rate_limit_headers.get("x-ratelimit-remaining-requests") == "0" and float(resp.rate_limit_headers.get("retry-after", "0")) > 60.0)
                if is_daily or attempt == max_retries:
                    # Do not retry daily budget or final attempt
                    break

                retries_count += 1
                backoff_sec = resp.retry_after_sec or compute_backoff(attempt)
                await asyncio.sleep(min(2.0, backoff_sec))

            d4 = (time.perf_counter() - t0) * 1000.0
            traces.append(PipelineTraceStep(
                step_name="4_answer_generation",
                started_at_ms=(t0 - t_global_start) * 1000.0,
                duration_ms=d4,
                status=resp.status if resp else "error",
                metadata={"retries": retries_count, "status": resp.status if resp else "none"}
            ))

            if resp.status != "ok":
                total_ms = (time.perf_counter() - t_global_start) * 1000.0
                await reliability_mgr.record_outcome(resp.status, retries=retries_count)
                return PipelineResult(
                    request_id=req_id,
                    question=request.question,
                    answer=f"Service Unavailable: {resp.error_message}",
                    refused=True,
                    refusal_reason=resp.error_message,
                    retrieved_passages=passages,
                    first_token_ms=None,
                    provider_ttft_ms=None,
                    total_ms=total_ms,
                    queue_wait_ms=queue_wait_ms,
                    retries=retries_count,
                    status=resp.status,
                    traces=traces,
                    source=resp.source,
                    retrieval_setting=request.retrieval_setting,
                    answer_grade=None
                )

            # -------------------------------------------------------------
            # Stage 5: Answer & Citation Verification
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            is_valid, valid_citations, refusal_reason, grade = verify_answer(
                resp.content,
                passages,
                is_expected_unanswerable=(understanding.complexity == "unanswerable")
            )
            d5 = (time.perf_counter() - t0) * 1000.0
            traces.append(PipelineTraceStep(
                step_name="5_answer_verification",
                started_at_ms=(t0 - t_global_start) * 1000.0,
                duration_ms=d5,
                status="ok" if is_valid else "failed",
                metadata={"is_valid": is_valid, "citations_count": len(valid_citations), "grade": grade}
            ))

            final_answer = resp.content
            refused = is_refusal(final_answer) or not is_valid

            if not is_valid and not refused:
                # If verifier detected missing/invalid citations, enforce safe refusal
                final_answer = (
                    "Based on the retrieved research papers, there is insufficient verified evidence to answer this question. "
                    "The generated claims could not be verified against the source citations."
                )
                refused = True
                grade = "wrong"

            total_ms = (time.perf_counter() - t_global_start) * 1000.0
            await reliability_mgr.record_outcome("ok", retries=retries_count)

            # End-to-end user-perceived TTFT from arrival to first token emitted
            user_ttft = ((t_winning_attempt_start - t_global_start) * 1000.0) + (resp.first_token_ms or 0.0)

            return PipelineResult(
                request_id=req_id,
                question=request.question,
                answer=final_answer,
                citations=valid_citations,
                refused=refused,
                refusal_reason=refusal_reason,
                retrieved_passages=passages,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                first_token_ms=round(user_ttft, 1),
                provider_ttft_ms=round(resp.first_token_ms, 1) if resp.first_token_ms else None,
                total_ms=total_ms,
                queue_wait_ms=queue_wait_ms,
                retries=retries_count,
                status="ok",
                traces=traces,
                source=resp.source,
                retrieval_setting=request.retrieval_setting,
                answer_grade=grade
            )

        finally:
            reliability_mgr.get_semaphore().release()

    async def execute_stream(
        self,
        request: AskRequest,
        request_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        SSE Streaming Pipeline execution.
        Emits stages, token deltas, and final verified response.
        """
        req_id = request_id or str(uuid.uuid4())
        t_global_start = time.perf_counter()
        
        # Concurrency & Queue wait tracking
        t_queue_start = time.perf_counter()
        await reliability_mgr.track_queue_enter()
        await reliability_mgr.get_semaphore().acquire()
        queue_wait_ms = (time.perf_counter() - t_queue_start) * 1000.0
        await reliability_mgr.track_queue_leave()

        try:
            # Stage 1: Question Understanding
            yield {"type": "stage", "stage": "question_understanding"}
            understanding = analyze_question(request.question)

            if not understanding.needs_retrieval:
                total_ms = (time.perf_counter() - t_global_start) * 1000.0
                await reliability_mgr.record_outcome("ok", retries=0)
                refusal = "Based on the research library, this question cannot be answered. The provided corpus contains only three machine learning papers and does not contain information on this topic."
                yield {"type": "delta", "delta": refusal}
                yield {
                    "type": "done",
                    "result": PipelineResult(
                        request_id=req_id,
                        question=request.question,
                        answer=refusal,
                        refused=True,
                        refusal_reason="Out of corpus scope",
                        first_token_ms=total_ms,
                        total_ms=total_ms,
                        queue_wait_ms=queue_wait_ms,
                        status="ok",
                        source=getattr(self.provider, "source", "stand-in"),
                        retrieval_setting=request.retrieval_setting,
                        answer_grade="refused_correctly"
                    ).model_dump()
                }
                return

            # Stage 2: Retrieval Tool
            yield {"type": "stage", "stage": "retrieval_tool"}
            tool_output = await self.retrieval_tool.execute(
                RetrievalToolInput(query=request.question, retrieval_setting=request.retrieval_setting)
            )
            passages = tool_output.passages

            # Stage 3: Decision & Branching
            yield {"type": "stage", "stage": "decision_and_branching"}
            if not passages or (passages[0].score < 0.1 and request.retrieval_setting != "C"):
                sec_output = await self.retrieval_tool.execute(
                    RetrievalToolInput(
                        query=f"{request.question} {' '.join(understanding.technical_terms)}".strip(),
                        retrieval_setting="C",
                        strategy="secondary_fallback"
                    )
                )
                if sec_output.passages:
                    passages = sec_output.passages

            if not passages:
                total_ms = (time.perf_counter() - t_global_start) * 1000.0
                await reliability_mgr.record_outcome("ok", retries=0)
                refusal = "Based on the retrieved research papers, there is insufficient evidence to answer this question. The library does not contain relevant passages."
                yield {"type": "delta", "delta": refusal}
                yield {
                    "type": "done",
                    "result": PipelineResult(
                        request_id=req_id,
                        question=request.question,
                        answer=refusal,
                        refused=True,
                        first_token_ms=total_ms,
                        total_ms=total_ms,
                        queue_wait_ms=queue_wait_ms,
                        status="ok",
                        source=getattr(self.provider, "source", "stand-in"),
                        retrieval_setting=request.retrieval_setting,
                        answer_grade="refused_correctly"
                    ).model_dump()
                }
                return

            # Stage 4: Generation with Internal Buffering
            yield {"type": "stage", "stage": "answer_generation"}
            context_blocks = [
                f"--- Passage from '{p.paper_title}' (arXiv {p.arxiv_id}{p.version}), Page {p.page_number}, Section: {p.section} ---\n{p.text}"
                for p in passages
            ]
            user_prompt = f"Question: {request.question}\n\nRetrieved Passages from Corpus:\n{chr(10).join(context_blocks)}\n\nAnswer:"

            full_content = ""
            final_resp: Optional[ProviderResponse] = None
            provider_first_token_ms: Optional[float] = None

            # Buffer complete LLM generation internally without streaming unverified tokens to the client
            async for chunk in self.provider.generate_stream(prompt=user_prompt, system_prompt=SYSTEM_PROMPT):
                if chunk["delta"]:
                    if provider_first_token_ms is None:
                        provider_first_token_ms = (time.perf_counter() - t_global_start) * 1000.0
                    full_content += chunk["delta"]
                if chunk["done"] and chunk["response"]:
                    final_resp = chunk["response"]

            if not final_resp or final_resp.status != "ok":
                total_ms = (time.perf_counter() - t_global_start) * 1000.0
                err_status = final_resp.status if final_resp else "error"
                await reliability_mgr.record_outcome(err_status, retries=0)
                yield {
                    "type": "done",
                    "result": PipelineResult(
                        request_id=req_id,
                        question=request.question,
                        answer=final_resp.error_message if final_resp else "Unknown error",
                        status=err_status,
                        first_token_ms=None,
                        provider_ttft_ms=None,
                        total_ms=total_ms,
                        queue_wait_ms=queue_wait_ms,
                        source=final_resp.source if final_resp else "stand-in",
                        retrieval_setting=request.retrieval_setting,
                        answer_grade=None
                    ).model_dump()
                }
                return

            # Stage 5: Answer & Citation Verification Runs on Buffered Response BEFORE Client Delivery
            yield {"type": "stage", "stage": "answer_verification"}
            is_valid, valid_citations, refusal_reason, grade = verify_answer(
                final_resp.content,
                passages,
                is_expected_unanswerable=(understanding.complexity == "unanswerable")
            )

            refused = is_refusal(final_resp.content) or not is_valid
            if not is_valid and not is_refusal(final_resp.content):
                # Unverified claims or invalid citations are replaced with a deterministic refusal
                verified_answer = (
                    "Based on the retrieved research papers, there is insufficient verified evidence to answer this question. "
                    "The generated claims could not be verified against the source citations."
                )
                refused = True
                grade = "wrong"
            else:
                verified_answer = final_resp.content

            # Stage 6: Stream ONLY Verified Answer Content to Client via SSE
            user_first_token_ms = (time.perf_counter() - t_global_start) * 1000.0

            # Stream verified content incrementally to client
            words = re.findall(r'\S+|\s+', verified_answer)
            for word in words:
                yield {"type": "delta", "delta": word}

            total_ms = (time.perf_counter() - t_global_start) * 1000.0
            await reliability_mgr.record_outcome("ok", retries=0)

            result = PipelineResult(
                request_id=req_id,
                question=request.question,
                answer=verified_answer,
                citations=valid_citations,
                refused=refused,
                refusal_reason=refusal_reason,
                retrieved_passages=passages,
                tokens_in=final_resp.tokens_in,
                tokens_out=final_resp.tokens_out,
                first_token_ms=round(user_first_token_ms, 1),
                provider_ttft_ms=round(final_resp.first_token_ms, 1) if final_resp.first_token_ms else (round(provider_first_token_ms, 1) if provider_first_token_ms else None),
                total_ms=total_ms,
                queue_wait_ms=queue_wait_ms,
                retries=0,
                status="ok",
                source=final_resp.source,
                retrieval_setting=request.retrieval_setting,
                answer_grade=grade
            )
            yield {"type": "done", "result": result.model_dump()}

        finally:
            reliability_mgr.get_semaphore().release()
