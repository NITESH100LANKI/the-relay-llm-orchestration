import os
import sys
import asyncio
import time

print("Step 1: Setting environment", flush=True)
os.environ["PYTHONPATH"] = r"d:\The Relay\the-relay"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("Step 2: Importing modules", flush=True)
from app.schemas.models import AskRequest
print("Step 2a: Imported AskRequest", flush=True)
from app.pipeline.orchestrator import PipelineOrchestrator
print("Step 2b: Imported PipelineOrchestrator", flush=True)

async def main():
    print("Step 3: Inside async main()", flush=True)
    orch = PipelineOrchestrator()
    print("Step 4: Orchestrator created", flush=True)
    req = AskRequest(question="What is the Mimi audio codec in Moshi?", retrieval_setting="B")
    print("Step 5: Calling orch.execute()...", flush=True)
    t0 = time.time()
    res = await orch.execute(req)
    t1 = time.time()
    print(f"Step 6: orch.execute() completed in {round((t1-t0)*1000, 1)}ms!", flush=True)
    print("Status:", res.status, flush=True)
    print("Grade:", res.answer_grade, flush=True)
    print("Answer:\n", res.answer, flush=True)
    print("Citations count:", len(res.citations), flush=True)
    for c in res.citations:
        print(f" - {c.paper_title}, Page {c.page_number}", flush=True)

if __name__ == "__main__":
    print("Step 2c: In __main__, about to asyncio.run", flush=True)
    asyncio.run(main())
