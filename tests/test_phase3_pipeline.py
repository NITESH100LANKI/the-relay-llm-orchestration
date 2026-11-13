"""
Unit and Integration Tests for Phase 3:
- Question understanding
- Explicit tool invocation
- Conditional branching / secondary retrieval loop
- Grounded answer generation
- Citation verification & hallucination prevention
- Refusal behavior
- SSE streaming response
- Pipeline traces & latency tracking
"""

import pytest
import asyncio
from app.schemas.models import AskRequest, RetrievedPassage, Citation
from app.pipeline.question_analyzer import analyze_question
from app.tools.retrieval_tool import RetrievalTool, RetrievalToolInput
from app.pipeline.verifier import verify_answer, parse_citations, is_refusal
from app.providers.simulator_provider import SimulatorProvider, SimulatorConfig
from app.pipeline.orchestrator import PipelineOrchestrator

@pytest.mark.asyncio
async def test_question_understanding():
    # Technical question
    u1 = analyze_question("What is the Mimi audio codec in Moshi?")
    assert u1.needs_retrieval is True
    assert "moshi" in u1.technical_terms or "mimi" in u1.technical_terms
    assert "Moshi: a speech-text foundation model for real-time dialogue" in u1.target_papers

    # Out of scope query
    u2 = analyze_question("What is the stock price of Apple today?")
    assert u2.needs_retrieval is False
    assert u2.complexity == "unanswerable"

@pytest.mark.asyncio
async def test_retrieval_tool_execution():
    tool = RetrievalTool()
    inp = RetrievalToolInput(query="Scaled Dot-Product Attention Softmax", retrieval_setting="B")
    out = await tool.execute(inp)
    assert out.tool_name == "retrieve_paper_passages"
    assert out.passages_count > 0
    assert len(out.passages) <= 5
    first = out.passages[0]
    assert first.paper_title != ""
    assert first.page_number > 0
    assert first.arxiv_id != ""

@pytest.mark.asyncio
async def test_citation_parsing_and_verification():
    sample_text = (
        "Multi-Head Attention allows the model to jointly attend to information at different positions.\n"
        "[Citation: Attention Is All You Need, arXiv 1706.03762v7, Page 4, Section 3.2]"
    )
    citations = parse_citations(sample_text)
    assert len(citations) == 1
    c = citations[0]
    assert c.paper_title == "Attention Is All You Need"
    assert c.arxiv_id == "1706.03762"
    assert c.version == "v7"
    assert c.page_number == 4

    # Verify against dummy passages
    passages = [
        RetrievedPassage(
            chunk_id="c1",
            paper_title="Attention Is All You Need",
            arxiv_id="1706.03762",
            version="v7",
            page_number=4,
            section="3.2",
            text="Multi-Head Attention allows the model...",
            score=0.9,
            rank=1
        )
    ]
    is_valid, valid_cits, reason, grade = verify_answer(sample_text, passages)
    assert is_valid is True
    assert len(valid_cits) == 1
    assert grade == "correct"

@pytest.mark.asyncio
async def test_hallucination_refusal():
    # Answer citing non-existent passage or page
    hallucinated_answer = (
        "The model costs $50,000 per GPU to train.\n"
        "[Citation: Dynamic Chunking, arXiv 2507.07955v2, Page 99, Section Cost]"
    )
    passages = [
        RetrievedPassage(
            chunk_id="c1",
            paper_title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
            arxiv_id="2507.07955",
            version="v2",
            page_number=4,
            section="Method",
            text="Dynamic chunking...",
            score=0.8,
            rank=1
        )
    ]
    is_valid, valid_cits, reason, grade = verify_answer(hallucinated_answer, passages)
    assert is_valid is False
    assert grade == "wrong"

@pytest.mark.asyncio
async def test_full_pipeline_orchestration():
    sim_cfg = SimulatorConfig(base_ttft_ms=50.0, token_generation_ms=1.0)
    provider = SimulatorProvider(sim_cfg)
    orchestrator = PipelineOrchestrator(provider=provider)

    req = AskRequest(question="What is the Mimi audio codec in Moshi?", retrieval_setting="B", stream=False)
    result = await orchestrator.execute(req)

    assert result.status == "ok"
    assert result.total_ms > 0
    assert result.first_token_ms > 0
    assert len(result.traces) >= 4
    step_names = [t.step_name for t in result.traces]
    assert "1_question_understanding" in step_names
    assert "2_retrieval_tool" in step_names
    assert "3_decision_and_branching" in step_names
    assert "4_answer_generation" in step_names
    assert "5_answer_verification" in step_names
    assert result.source == "stand-in"
    assert len(result.citations) > 0 or result.refused

@pytest.mark.asyncio
async def test_out_of_scope_abstention():
    sim_cfg = SimulatorConfig(base_ttft_ms=20.0, token_generation_ms=1.0)
    provider = SimulatorProvider(sim_cfg)
    orchestrator = PipelineOrchestrator(provider=provider)

    req = AskRequest(question="What is the stock price of Tesla or the hardware purchase budget?", retrieval_setting="B")
    result = await orchestrator.execute(req)
    assert result.refused is True
    assert result.answer_grade == "refused_correctly"

