"""
Interactive and Scriptable Command-Line Interface for The Relay.
Usage:
    python -m app.cli --question "What is the Mimi audio codec in Moshi?" --setting B --no-stream
    python -m app.cli "What is the Mimi audio codec in Moshi?"
"""

import sys
import argparse
import asyncio
import time
from app.schemas.models import AskRequest
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.groq_provider import GroqProvider
from app.providers.simulator_provider import SimulatorProvider
from app.core.config import settings

async def run_query(question: str, setting: str = "B", stream: bool = False, provider_mode: str = "stand-in"):
    print(f"\n=======================================================", flush=True)
    print(f" THE RELAY RESEARCH ASSISTANT — QUERY ENGINE", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Question:  {question}", flush=True)
    print(f"Setting:   Config {setting}", flush=True)
    print(f"Mode:      {provider_mode}", flush=True)
    print(f"Streaming: {stream}", flush=True)
    print(f"-------------------------------------------------------\n", flush=True)

    if provider_mode == "real":
        provider = GroqProvider()
    else:
        provider = SimulatorProvider()

    orchestrator = PipelineOrchestrator(provider=provider)
    req = AskRequest(question=question, retrieval_setting=setting, stream=stream)

    if stream:
        print("[Streaming Response Start]", flush=True)
        async for event in orchestrator.execute_stream(req):
            if event["type"] == "stage":
                print(f"\n[Stage] -> {event['stage']}", flush=True)
            elif event["type"] == "delta":
                sys.stdout.write(event["delta"])
                sys.stdout.flush()
            elif event["type"] == "done":
                res = event["result"]
                print(f"\n\n-------------------------------------------------------", flush=True)
                print(f"[Telemetry]", flush=True)
                print(f"TTFT:        {res['first_token_ms']:.1f} ms", flush=True)
                print(f"Total Time:  {res['total_ms']:.1f} ms", flush=True)
                print(f"Tokens In:   {res['tokens_in']}", flush=True)
                print(f"Tokens Out:  {res['tokens_out']}", flush=True)
                print(f"Status:      {res['status']}", flush=True)
                print(f"Source:      {res['source']}", flush=True)
                print(f"Grade:       {res.get('answer_grade')}", flush=True)
                if res.get("citations"):
                    print(f"\n[Verified Citations ({len(res['citations'])})]", flush=True)
                    for c in res["citations"]:
                        print(f" - {c['paper_title']} (arXiv {c['arxiv_id']}{c['version']}), Page {c['page_number']}, Section: {c.get('section')}", flush=True)
    else:
        result = await orchestrator.execute(req)
        print(f"\n[Answer]:\n{result.answer}\n", flush=True)
        print(f"-------------------------------------------------------", flush=True)
        print(f"[Telemetry]", flush=True)
        print(f"TTFT:        {result.first_token_ms:.1f} ms", flush=True)
        print(f"Total Time:  {result.total_ms:.1f} ms", flush=True)
        print(f"Queue Wait:  {result.queue_wait_ms:.1f} ms", flush=True)
        print(f"Tokens In:   {result.tokens_in}", flush=True)
        print(f"Tokens Out:  {result.tokens_out}", flush=True)
        print(f"Status:      {result.status}", flush=True)
        print(f"Source:      {result.source}", flush=True)
        print(f"Grade:       {result.answer_grade}", flush=True)
        if result.citations:
            print(f"\n[Verified Citations ({len(result.citations)})]", flush=True)
            for c in result.citations:
                print(f" - {c.paper_title} (arXiv {c.arxiv_id}{c.version}), Page {c.page_number}, Section: {c.section}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="The Relay CLI")
    parser.add_argument("pos_question", nargs="?", default=None, help="Question as positional argument.")
    parser.add_argument("--question", "-q", default=None, help="Question as named option.")
    parser.add_argument("--setting", choices=["A", "B", "C"], default="B", help="Retrieval preset (A, B, C).")
    parser.add_argument("--stream", action="store_true", default=False, help="Enable token streaming.")
    parser.add_argument("--provider", choices=["real", "stand-in"], default="stand-in", help="LLM provider override.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt loop.")
    args = parser.parse_args()

    q = args.question or args.pos_question
    mode = args.provider or settings.provider_mode

    if not q:
        if args.interactive and sys.stdin.isatty():
            print("Welcome to The Relay Interactive CLI (Type 'exit' or 'quit' to stop)")
            while True:
                try:
                    line = input("\nEnter your question: ").strip()
                    if not line or line.lower() in ["exit", "quit"]:
                        break
                    asyncio.run(run_query(line, setting=args.setting, stream=args.stream, provider_mode=mode))
                except (KeyboardInterrupt, EOFError):
                    break
            return
        else:
            q = "What is Scaled Dot-Product Attention in the Transformer?"
            print(f"[CLI] No question specified. Using default research question: '{q}'\n")

    asyncio.run(run_query(q, setting=args.setting, stream=args.stream, provider_mode=mode))

if __name__ == "__main__":
    main()
