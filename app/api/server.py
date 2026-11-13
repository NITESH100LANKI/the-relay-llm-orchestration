"""
FastAPI Server for The Relay.
Exposes:
- POST /ask (Supports streaming SSE and JSON response)
- GET /health
- GET /metrics
- GET /config
- GET / (Interactive Web UI for live querying and streaming observation)
"""

import json
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.models import AskRequest, AskResponse, PipelineResult
from app.pipeline.orchestrator import PipelineOrchestrator
from app.core.config import settings
from app.core.reliability import reliability_mgr
from app.providers.groq_provider import GroqProvider
from app.providers.simulator_provider import SimulatorProvider

app = FastAPI(
    title="The Relay",
    description="Production-Grade LLM Orchestration, RAG & Performance Engineering System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = PipelineOrchestrator()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "provider_mode": settings.provider_mode,
        "active_model": settings.groq_model if settings.provider_mode == "real" else "Stand-in Calibrated Simulator",
        "current_queue_length": reliability_mgr.current_queue_length,
        "circuit_breaker": reliability_mgr.circuit_breaker.state
    }

@app.get("/metrics")
async def get_metrics():
    return reliability_mgr.get_metrics()

@app.get("/config")
async def get_config():
    return {
        "provider_mode": settings.provider_mode,
        "model": settings.groq_model,
        "max_concurrency": settings.max_concurrency,
        "rate_limit_rpm": settings.rate_limit_rpm,
        "rate_limit_tpm": settings.rate_limit_tpm,
        "rate_limit_rpd": settings.rate_limit_rpd,
        "request_deadline_sec": settings.request_deadline_sec,
        "corpus_papers": [
            "Attention Is All You Need (arXiv 1706.03762v7)",
            "Moshi: a speech-text foundation model for real-time dialogue (arXiv 2410.00037v1)",
            "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (arXiv 2507.07955v2)"
        ],
        "retrieval_presets": {
            "A": {"chunk_size": 300, "overlap": 50, "top_k": 3, "reranking": False},
            "B": {"chunk_size": 500, "overlap": 100, "top_k": 5, "reranking": False},
            "C": {"chunk_size": 500, "overlap": 100, "top_k": 5, "reranking": True}
        }
    }

@app.post("/ask")
async def ask_endpoint(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if request.stream:
        async def event_generator():
            async for event in orchestrator.execute_stream(request):
                yield f"data: {json.dumps(event)}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Non-streaming request
    try:
        deadline = request.request_deadline_ms / 1000.0 if request.request_deadline_ms else settings.request_deadline_sec
        result = await asyncio.wait_for(
            orchestrator.execute(request),
            timeout=deadline
        )
        return AskResponse(
            request_id=result.request_id,
            question=result.question,
            answer=result.answer,
            citations=result.citations,
            refused=result.refused,
            retrieval_setting=result.retrieval_setting,
            first_token_ms=result.first_token_ms,
            total_ms=result.total_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            retries=result.retries,
            status=result.status,
            source=result.source
        )
    except asyncio.TimeoutError:
        await reliability_mgr.record_outcome("timeout", retries=0)
        raise HTTPException(status_code=504, detail=f"Request timed out after {deadline}s deadline.")

@app.get("/", response_class=HTMLResponse)
async def interactive_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Relay — Research Assistant</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }
        .container { max-width: 900px; margin: 0 auto; background: #1e293b; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        h1 { color: #38bdf8; margin-top: 0; }
        label { display: block; margin-top: 1rem; font-weight: 600; color: #94a3b8; }
        textarea, select, button { width: 100%; padding: 0.75rem; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #fff; font-size: 1rem; box-sizing: border-box; }
        textarea { height: 100px; resize: vertical; margin-top: 0.5rem; }
        button { background: #0284c7; color: white; border: none; font-weight: bold; cursor: pointer; margin-top: 1.5rem; transition: background 0.2s; }
        button:hover { background: #0369a1; }
        .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.5rem; }
        #output { margin-top: 2rem; padding: 1.5rem; background: #090d16; border-radius: 8px; border: 1px solid #334155; min-height: 120px; white-space: pre-wrap; line-height: 1.6; }
        .citation { color: #38bdf8; background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
        .telemetry { display: flex; gap: 1.5rem; margin-top: 1rem; font-size: 0.85em; color: #94a3b8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>The Relay — Engineering Research Assistant</h1>
        <p style="color: #94a3b8;">Grounded Question-Answering across Attention Is All You Need, Moshi, and Dynamic Chunking.</p>
        
        <label for="question">Your Question:</label>
        <textarea id="question" placeholder="e.g. What is Scaled Dot-Product Attention in the Transformer?">What is Scaled Dot-Product Attention in the Transformer?</textarea>
        
        <div class="meta-grid">
            <div>
                <label for="setting">Retrieval Configuration:</label>
                <select id="setting">
                    <option value="B" selected>Config B (500 tokens, 100 overlap, Top 5)</option>
                    <option value="A">Config A (300 tokens, 50 overlap, Top 3)</option>
                    <option value="C">Config C (500 tokens, 100 overlap, Top 5 + Rerank)</option>
                </select>
            </div>
            <div>
                <label for="stream">Response Mode:</label>
                <select id="stream">
                    <option value="true" selected>Streaming (SSE)</option>
                    <option value="false">Non-Streaming</option>
                </select>
            </div>
        </div>

        <button onclick="submitQuery()">Submit Query</button>

        <div class="telemetry" id="telemetry">
            <span>TTFT: -</span>
            <span>Total Time: -</span>
            <span>Status: Ready</span>
            <span>Source: -</span>
        </div>

        <div id="output">Answer will appear here...</div>
    </div>

    <script>
        async function submitQuery() {
            const question = document.getElementById('question').value;
            const setting = document.getElementById('setting').value;
            const stream = document.getElementById('stream').value === "true";
            const outDiv = document.getElementById('output');
            const telemDiv = document.getElementById('telemetry');
            
            outDiv.textContent = "Connecting to pipeline...";
            telemDiv.innerHTML = "<span>Status: Running...</span>";
            const t0 = performance.now();

            if (stream) {
                outDiv.textContent = "";
                const resp = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ question, retrieval_setting: setting, stream: true })
                });
                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let firstTokenMs = null;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.substring(6));
                            if (data.type === 'delta') {
                                if (firstTokenMs === null) {
                                    firstTokenMs = Math.round(performance.now() - t0);
                                }
                                outDiv.textContent += data.delta;
                            } else if (data.type === 'done') {
                                const totalMs = Math.round(performance.now() - t0);
                                const res = data.result;
                                telemDiv.innerHTML = `<span>TTFT: ${res.first_token_ms.toFixed(1)}ms</span> <span>Total: ${totalMs}ms</span> <span>Status: ${res.status}</span> <span>Source: ${res.source}</span>`;
                            }
                        }
                    }
                }
            } else {
                const resp = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ question, retrieval_setting: setting, stream: false })
                });
                const data = await resp.json();
                outDiv.textContent = data.answer;
                telemDiv.innerHTML = `<span>TTFT: ${data.first_token_ms.toFixed(1)}ms</span> <span>Total: ${data.total_ms.toFixed(1)}ms</span> <span>Status: ${data.status}</span> <span>Source: ${data.source}</span>`;
            }
        }
    </script>
</body>
</html>
"""

