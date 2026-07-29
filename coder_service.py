"""
Coder Service — FastAPI wrapper around coding_agent.py
=====================================================================
Exposes coding_agent.py's existing solve() over HTTP so Coordinator
(running as a separate process/console) can hand it a task and get a
result back, instead of importing coding_agent.py's internals
directly.

coding_agent.py itself is UNTOUCHED -- this file imports its solve()
and AgentConfig and nothing else, so coding_agent.py keeps working
exactly as before via its own standalone CLI.

Every request/response is keyed by a `task_id` supplied by the
CALLER (Coordinator generates it, not this service) -- this service
just uses that id for its own bookkeeping (the `_jobs` dict below) and
echoes it back, so it's a stable identifier the rest of the system
(Critic, Memory, ...) can key off later, even though today every call
is a single blocking request/response.

Calls are BLOCKING by design right now: POST /solve does not return
until coding_agent.py's solve() has finished (including any
--verify-tests-style interactive prompts, which appear IN THIS
SERVICE's terminal, not Coordinator's -- watch this terminal during a
run if verify_tests=True is sent).

Run it:
    uvicorn coder_service:app --port 8001

Endpoints:
    POST /solve         -> run solve() for a task_id, blocks until done
    GET  /tasks/{id}     -> inspect a previously-run task's stored result
    GET  /health         -> liveness check
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from coding_agent import solve, AgentConfig


app = FastAPI(title="Coder Service")

# In-memory bookkeeping keyed by task_id. Not persisted -- restarting
# this service loses history. Fine for now; a real store (or Memory
# itself) is the natural upgrade if this needs to survive restarts.
_jobs: dict = {}


class SolveRequest(BaseModel):
    task_id: str
    task: str
    backend: str = "mock"
    ollama_model: str = "llama3"
    ollama_url: str = "http://localhost:11434"
    max_iters: int = 20
    n_candidates: int = 8
    verify_tests: bool = False  # kept fully functional per current design -- see module docstring
    auto_verify_after_rounds: int = 5  # 0 disables; see AgentConfig docstring in coding_agent.py


class SolveResponse(BaseModel):
    task_id: str
    status: str                 # "done" | "error"
    final_code: Optional[str] = None
    score: Optional[float] = None
    rounds_used: Optional[int] = None
    error: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve_task(req: SolveRequest):
    if req.task_id in _jobs:
        raise HTTPException(
            status_code=409,
            detail=f"task_id '{req.task_id}' was already submitted "
                    "(each task_id must be used once).",
        )

    _jobs[req.task_id] = {
        "status": "running",
        "task": req.task,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    cfg = AgentConfig(
        model=req.backend,
        ollama_model=req.ollama_model,
        ollama_url=req.ollama_url,
        max_iters=req.max_iters,
        n_candidates=req.n_candidates,
        verify_tests=req.verify_tests,
        auto_verify_after_rounds=req.auto_verify_after_rounds,
    )

    try:
        print(f"\n[coder_service] starting task_id={req.task_id}: {req.task[:80]}")
        outcome = solve(req.task, cfg)
    except Exception as e:
        _jobs[req.task_id]["status"] = "error"
        _jobs[req.task_id]["error"] = str(e)
        return SolveResponse(task_id=req.task_id, status="error", error=str(e))

    result = SolveResponse(
        task_id=req.task_id,
        status="done",
        final_code=outcome["final_code"],
        score=outcome["score"],
        rounds_used=len(outcome["history"]),
    )
    _jobs[req.task_id].update({
        "status": "done",
        "final_code": outcome["final_code"],
        "score": outcome["score"],
        "rounds_used": len(outcome["history"]),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[coder_service] finished task_id={req.task_id}: score={outcome['score']:.0%}")
    return result


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    if task_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"No task with id '{task_id}'.")
    return _jobs[task_id]
