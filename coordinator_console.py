"""
Coordinator Console — the main user-facing entry point
=====================================================================
This is what the user actually runs and types into. It loops: ask for
a task, decide locally (via coordinator.py's decide(), unchanged, no
network call for THIS part -- Coordinator's own LLM assessment stays
local to this process), then either stop (route == planner, not wired
up yet) or hand the task off to the Coder Service over HTTP.

task_id is generated HERE, by Coordinator, before the Coder Service
ever sees the task -- it's the shared identifier the rest of the
system (Critic, Memory, ...) will key off later as they're turned
into services too. This process is the only one that mints new ids.

Calls to the Coder Service are BLOCKING: this console just waits for
the HTTP response. If verify_tests is on, the interactive prompts
happen in the CODER SERVICE's terminal, not here -- watch that
terminal during a run.

Requires the Coder Service to already be running (see coder_service.py
-- `uvicorn coder_service:app --port 8001`, or just use run_all.py
which starts both).
"""

import sys
import uuid
import requests

import coordinator


CODER_SERVICE_URL = "http://localhost:8001"
PLAN_CONTROLLER_URL = "http://localhost:8002"


def _check_service_up(url: str) -> bool:
    try:
        resp = requests.get(f"{url}/health", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def dispatch_to_plan_controller(goal: str, task_id: str, backend: str, ollama_model: str,
                                 ollama_url: str, plan_url: str = PLAN_CONTROLLER_URL) -> dict:
    """POST the goal to Plan Controller and block until it responds.
    NOTE: if issues are found and skip_review is left False (default),
    Plan Controller's human review prompts appear in PLAN CONTROLLER'S
    terminal, not here -- watch that terminal during a run."""
    payload = {
        "task_id": task_id,
        "goal": goal,
        "backend": backend,
        "ollama_model": ollama_model,
        "ollama_url": ollama_url,
        "skip_review": False,
    }
    print(f"\n[coordinator] dispatching task_id={task_id} to Plan Controller at {plan_url} ...")
    print("[coordinator] if issues are found, check Plan Controller's terminal "
          "for the review prompts.")

    resp = requests.post(f"{plan_url}/plan", json=payload, timeout=None)
    resp.raise_for_status()
    return resp.json()


def dispatch_to_coder(task: str, task_id: str, backend: str, ollama_model: str,
                       ollama_url: str, verify_tests: bool, coder_url: str = CODER_SERVICE_URL) -> dict:
    """POST the task to the Coder Service and block until it responds."""
    payload = {
        "task_id": task_id,
        "task": task,
        "backend": backend,
        "ollama_model": ollama_model,
        "ollama_url": ollama_url,
        "verify_tests": verify_tests,
    }
    print(f"\n[coordinator] dispatching task_id={task_id} to Coder Service at {coder_url} ...")
    if verify_tests:
        print("[coordinator] --verify-tests is on -- check the Coder Service's "
              "terminal for the test-verification prompts.")

    resp = requests.post(f"{coder_url}/solve", json=payload, timeout=None)
    resp.raise_for_status()
    return resp.json()


def run_console(backend: str, ollama_model: str, ollama_url: str,
                 verify_tests: bool, coder_url: str = CODER_SERVICE_URL,
                 plan_url: str = PLAN_CONTROLLER_URL):
    print("=== Coordinator Console ===")
    print(f"Coder Service: {coder_url}")
    print(f"Plan Controller: {plan_url}")
    if not _check_service_up(coder_url):
        print(f"\nWarning: could not reach the Coder Service at {coder_url}.")
        print("Start it first: uvicorn coder_service:app --port 8001")
        print("(or use run_all.py to start all services at once)\n")
    if not _check_service_up(plan_url):
        print(f"\nWarning: could not reach Plan Controller at {plan_url}.")
        print("Start it first: uvicorn plan_controller:app --port 8002")
        print("(or use run_all.py to start all services at once)\n")

    coord_cfg = coordinator.CoordinatorConfig(
        model=backend, ollama_model=ollama_model, ollama_url=ollama_url,
    )

    print("\nType a task, or 'exit'/'quit' to stop.\n")

    while True:
        try:
            task = input("task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not task:
            continue
        if task.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        try:
            decision = coordinator.decide(task, coord_cfg)
        except (RuntimeError, ValueError) as e:
            print(f"Error (coordinator): {e}\n")
            continue

        decision.pretty_print()

        task_id = str(uuid.uuid4())

        if decision.route == "planner":
            try:
                result = dispatch_to_plan_controller(
                    task, task_id, backend, ollama_model, ollama_url, plan_url,
                )
            except requests.exceptions.ConnectionError:
                print(f"\nCould not reach Plan Controller at {plan_url}. "
                      "Is it running?\n")
                continue
            except requests.exceptions.HTTPError as e:
                print(f"\nPlan Controller returned an error: {e}\n")
                continue

            if result["status"] == "error":
                print(f"\n[task_id={task_id}] Plan Controller failed: {result['error']}\n")
                continue

            plan = result["plan"]
            print(f"\n[coordinator] received final plan back from Plan Controller "
                  f"(task_id={task_id})")
            print(f"=== Plan ready [task_id={task_id}] ===")
            print(f"  {len(plan['tasks'])} subtask(s), "
                  f"{len(result['issues'])} issue(s) flagged during review:")
            for t in plan["tasks"]:
                deps = f" (depends on: {', '.join(t['depends_on'])})" if t["depends_on"] else ""
                print(f"  [{t['id']}] {t['class_name']}.{t['function_name']}"
                      f"({', '.join(t['params'])}){deps}")
                print(f"      \"{t['description']}\"")

            print(f"\n=== Full Plan Summary [task_id={task_id}] (for Memory) ===")
            print(f"  type: plan")
            print(f"  generated_by: coordinator -> planner + plan_controller")
            print(f"  original goal: \"{task}\"")
            print(f"\n  subtasks:")
            for t in plan["tasks"]:
                deps = ", ".join(t["depends_on"]) if t["depends_on"] else "(none)"
                print(f"    task_id_local={t['id']}  class_id={t['class_id']}  "
                      f"class_name={t['class_name']}  function_name={t['function_name']}")
                print(f"      description: \"{t['description']}\"")
                print(f"      params: {t['params']}  returns: {t['returns']}  depends_on: {deps}")
            print(f"\n  classes:")
            for c in plan["classes"]:
                print(f"    class_id={c['class_id']}  class_name={c['class_name']}  "
                      f"task_ids={c['task_ids']}")

            # Coder is not wired in for individual subtasks yet -- that's a later step.
            print()
            continue

        try:
            result = dispatch_to_coder(
                task, task_id, backend, ollama_model, ollama_url, verify_tests, coder_url,
            )
        except requests.exceptions.ConnectionError:
            print(f"\nCould not reach the Coder Service at {coder_url}. "
                  "Is it running?\n")
            continue
        except requests.exceptions.HTTPError as e:
            print(f"\nCoder Service returned an error: {e}\n")
            continue

        if result["status"] == "error":
            print(f"\n[task_id={task_id}] Coder Service failed: {result['error']}\n")
            continue

        print(f"\n[coordinator] received final result back from Coder Service "
              f"(task_id={task_id})")
        print(f"=== Task done [task_id={task_id}] ===")
        print(f"  score: {result['score']:.0%}")
        print(f"  rounds used: {result['rounds_used']}")
        print(f"  final code:\n{result['final_code']}\n")


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Coordinator Console: the main entry point")
    p.add_argument("--backend", choices=["mock", "ollama", "openai", "anthropic"], default="mock")
    p.add_argument("--ollama-model", default="llama3")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--verify-tests", action="store_true",
                    help="Ask the Coder Service to run human-in-the-loop test "
                         "verification -- prompts will appear in the Coder "
                         "Service's terminal, not here.")
    p.add_argument("--coder-url", default=CODER_SERVICE_URL)
    p.add_argument("--plan-url", default=PLAN_CONTROLLER_URL)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_console(
        backend=args.backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        verify_tests=args.verify_tests,
        coder_url=args.coder_url,
        plan_url=args.plan_url,
    )
