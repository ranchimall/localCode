"""
Plan Controller — FastAPI service wrapping planner.py, with the
issue-detection + human-review logic that used to live in a separate
plan_fixer.py now folded directly into this file (no plan_fixer.py
anymore -- Plan Controller absorbs that role entirely).

Flow (this service, one HTTP request):
  POST /plan {goal, ...}
    -> planner.plan(goal, cfg)        [imported from planner.py, in-process]
       -> a TaskDAG (the subtask breakdown) comes back
    -> detect_issues(goal, dag)       [ported from plan_fixer.py]
       -> a list of heuristic-flagged issues
    -> interactive_review(...)        [ported from plan_fixer.py]
       -> human reviews/edits issues ONE AT A TIME, nothing auto-applied
    -> final, re-validated TaskDAG returned as JSON in the response

planner.py itself is UNTOUCHED -- this file imports plan(), TaskDAG,
Task, PlannerConfig from it and nothing else, same relationship
coder_service.py has with coding_agent.py.

IMPORTANT -- same caveat as Coder Service's --verify-tests: the human
review step below calls Python's input(), which will prompt IN THIS
SERVICE'S TERMINAL, not Coordinator Console's. Watch this terminal
during a run if a task routes to "planner" and issues are found.

Run it:
    uvicorn plan_controller:app --port 8002

Endpoints:
    POST /plan           -> full plan-and-review flow, blocks until done
    GET  /jobs/{task_id} -> inspect a previously-run plan's stored result
    GET  /health          -> liveness check
"""

import re
import json
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from planner import Task, TaskDAG, PlannerConfig, plan as planner_plan


app = FastAPI(title="Plan Controller")

# In-memory bookkeeping keyed by task_id, same pattern as coder_service.py.
_jobs: dict = {}


# ----------------------------------------------------------------------
# Issue -- ported from plan_fixer.py, unchanged
# ----------------------------------------------------------------------
class Issue:
    def __init__(self, kind: str, message: str, task_ids: List[str], detail: str = ""):
        self.kind = kind
        self.message = message
        self.task_ids = task_ids
        self.detail = detail

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message,
                "task_ids": self.task_ids, "detail": self.detail}


# ----------------------------------------------------------------------
# Detection -- heuristics only, ported verbatim from plan_fixer.py.
# Every function here is a "maybe check this" signal for the human,
# never a verdict.
# ----------------------------------------------------------------------
_ACTOR_PAIR_RE = re.compile(
    r"([A-Za-z][A-Za-z ]{2,30}?)\s+and\s+([A-Za-z][A-Za-z ]{2,30}?)\s+"
    r"(?:can|will|must|shall|may|are|is)\b", re.IGNORECASE
)

_ACTION_VERB_RE = re.compile(
    r"\b(invoke|invokes|invoking|claim|claims|claiming|access|accesses|"
    r"accessing|receive|receives|receiving|take|takes|taking)\b", re.IGNORECASE
)

_LANGUAGE_KEYWORDS = {
    "javascript": "JavaScript", "js-based": "JavaScript", "typescript": "TypeScript",
    "python": "Python", "java": "Java", "golang": "Go", " go ": "Go",
    "rust": "Rust", "c++": "C++", "c#": "C#", "ruby": "Ruby", "php": "PHP",
}

_PERCENT_RE = re.compile(r"\b(\d{1,3})\s*(?:percent|%)", re.IGNORECASE)

_GENERIC_PIPELINE_VERBS = {"read", "extract", "deduplicate", "dedupe", "parse",
                            "fetch", "load", "scrape", "crawl", "save", "store"}


def _entity_coverage_issues(goal: str, tasks: List[Task]) -> List[Issue]:
    issues = []
    task_text = {t.id: f"{t.description} {t.function_name}".lower() for t in tasks}

    for m in _ACTOR_PAIR_RE.finditer(goal):
        actor_a, actor_b = m.group(1).strip().lower(), m.group(2).strip().lower()
        tail = goal[m.end(): m.end() + 120]
        verb_match = _ACTION_VERB_RE.search(tail)
        if not verb_match:
            continue
        verb_stem = verb_match.group(1).lower()[:5]

        a_key = actor_a.split()[-1][:5]
        b_key = actor_b.split()[-1][:5]

        for tid, text in task_text.items():
            if verb_stem not in text:
                continue
            has_a, has_b = a_key in text, b_key in text
            if has_a != has_b:
                missing_actor = actor_b if has_a else actor_a
                issues.append(Issue(
                    kind="entity_coverage",
                    message=(f"Task {tid} mentions '{verb_match.group(1)}' but its "
                             f"description only names one of '{actor_a}' / '{actor_b}' "
                             f"-- possibly dropped '{missing_actor}'."),
                    task_ids=[tid],
                    detail=f"Goal clause: \"...{actor_a} and {actor_b} {tail.strip()[:80]}...\"",
                ))
    return issues


def _missing_dependency_issues(tasks: List[Task]) -> List[Issue]:
    issues = []
    by_id = {t.id: t for t in tasks}

    def transitive_deps(tid, seen=None):
        seen = seen or set()
        for dep in by_id[tid].depends_on:
            if dep not in seen:
                seen.add(dep)
                transitive_deps(dep, seen)
        return seen

    producer_re = re.compile(r"calculate|compute|generate|produce", re.I)
    producers = [t for t in tasks if producer_re.search(t.function_name)]
    producer_ids = {t.id for t in producers}

    def modifier_words(fn_name: str) -> List[str]:
        return [w.lower() for w in re.findall(r"[A-Z][a-z]+", fn_name)]

    producer_modifiers = {p.id: modifier_words(p.function_name) for p in producers}

    for producer in producers:
        mods = producer_modifiers[producer.id]
        unique = [w for w in mods if len(w) >= 4 and
                  sum(1 for other in producers if w in producer_modifiers[other.id]) == 1]
        match_words = unique or mods
        if not match_words:
            continue

        for consumer in tasks:
            if consumer.id == producer.id or consumer.id in producer_ids:
                continue
            text = f"{consumer.description} {consumer.function_name}".lower()
            if not any(w in text for w in match_words):
                continue
            consumer_deps = transitive_deps(consumer.id) | {consumer.id}
            producer_deps = transitive_deps(producer.id) | {producer.id}
            if producer.id in consumer_deps:
                continue
            if consumer.id in producer_deps:
                continue
            hit = next(w for w in match_words if w in text)
            issues.append(Issue(
                kind="missing_dependency",
                message=(f"Task {consumer.id} references '{hit}' which "
                         f"{producer.id} ({producer.function_name}) produces, "
                         f"but {consumer.id} doesn't depend on {producer.id}."),
                task_ids=[consumer.id, producer.id],
                detail="Name-matching heuristic -- confirm this is a real ordering requirement.",
            ))
    return issues


def _language_mismatch_issues(goal: str) -> List[Issue]:
    goal_lower = f" {goal.lower()} "
    hits = []
    for kw, name in _LANGUAGE_KEYWORDS.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(kw.strip()) + r"(?![a-z0-9])", goal_lower):
            hits.append((kw.strip(), name))
    found = []
    for kw, name in hits:
        if not any(kw != other_kw and kw in other_kw for other_kw, _ in hits):
            found.append(name)
    found = sorted(set(found))
    if found:
        return [Issue(
            kind="language_mismatch",
            message=(f"Goal names target language(s) {found}, but this "
                     f"planner always emits Python-style `def f(self, ...)` signatures. "
                     f"Signatures will need manual translation for a Worker step."),
            task_ids=[],
        )]
    return []


def _uncaptured_value_issues(goal: str, tasks: List[Task]) -> List[Issue]:
    numbers = sorted(set(_PERCENT_RE.findall(goal)))
    if not numbers:
        return []
    all_text = " ".join(f"{t.description} {' '.join(t.params)} {t.returns}" for t in tasks)
    missing = [n for n in numbers if n not in all_text]
    if missing:
        return [Issue(
            kind="uncaptured_value",
            message=(f"Goal specifies percentage value(s) {missing} that don't appear "
                     f"in any task description/params/returns -- they may only exist "
                     f"implicitly, with nowhere for a Worker step to read them from."),
            task_ids=[],
        )]
    return []


def _split_words(name: str) -> List[str]:
    words = []
    for part in re.split(r"[_\W]+", name):
        words += re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", part)
    return [w.lower() for w in words if w]


def _scope_drift_issues(goal: str, tasks: List[Task]) -> List[Issue]:
    goal_lower = goal.lower()
    goal_words = set(re.findall(r"[a-z]+", goal_lower))

    def verb_supported_by_goal(verb: str) -> bool:
        stem = verb[:5]
        return any(w[:5] == stem for w in goal_words if len(w) >= 4)

    generic_in_goal = {v for v in _GENERIC_PIPELINE_VERBS if verb_supported_by_goal(v)}

    flagged_verbs = set()
    flagged_ids = []
    for t in tasks:
        words = _split_words(t.function_name)
        if not words:
            continue
        verb = words[0]
        if verb in _GENERIC_PIPELINE_VERBS and verb not in generic_in_goal:
            flagged_verbs.add(verb)
            flagged_ids.append(t.id)

    if not flagged_ids:
        return []
    return [Issue(
        kind="scope_drift",
        message=(f"{len(flagged_ids)} task(s) use generic pipeline verb(s) "
                 f"{sorted(flagged_verbs)} (read/extract/dedupe/parse/fetch/load-style), "
                 f"but none of these verbs appear anywhere in the goal text. This can "
                 f"happen when the planner LLM defaults to a generic data-pipeline "
                 f"template instead of the workflow actually described. Worth checking "
                 f"whether this plan reflects the goal at all, not just fixing individual tasks."),
        task_ids=flagged_ids,
        detail=f"Affected tasks: {flagged_ids}",
    )]


def detect_issues(goal: str, dag: TaskDAG) -> List[Issue]:
    tasks = list(dag.tasks.values())
    issues = []
    issues += _entity_coverage_issues(goal, tasks)
    issues += _missing_dependency_issues(tasks)
    issues += _language_mismatch_issues(goal)
    issues += _uncaptured_value_issues(goal, tasks)
    issues += _scope_drift_issues(goal, tasks)
    return issues


# ----------------------------------------------------------------------
# Interactive review -- ported from plan_fixer.py, unchanged. Nothing
# here is applied without an explicit human choice at the prompt.
# Runs in THIS SERVICE'S terminal -- see module docstring.
# ----------------------------------------------------------------------
def _prompt(msg: str) -> str:
    return input(msg).strip()


def _print_task(dag: TaskDAG, tid: str):
    t = dag.tasks[tid]
    deps = ", ".join(t.depends_on) if t.depends_on else "(none)"
    print(f"    [{t.id}] {t.class_name}.{t.function_name}({', '.join(t.params)})  depends_on: {deps}")
    print(f"        \"{t.description}\"")


def _edit_description(dag: TaskDAG, tid: str):
    print(f"    current: \"{dag.tasks[tid].description}\"")
    new_desc = _prompt("    new description (blank = leave unchanged): ")
    if new_desc:
        dag.tasks[tid].description = new_desc
        print("    updated.")


def _edit_dependencies(dag: TaskDAG, tid: str):
    print(f"    current depends_on: {dag.tasks[tid].depends_on}")
    raw = _prompt("    new depends_on, comma-separated task ids (blank = leave unchanged): ")
    if not raw:
        return
    new_deps = [d.strip() for d in raw.split(",") if d.strip()]
    unknown = [d for d in new_deps if d not in dag.tasks]
    if unknown:
        print(f"    ! unknown task id(s) {unknown}, not applying.")
        return
    dag.tasks[tid].depends_on = new_deps
    print("    updated.")


def _add_task(dag: TaskDAG):
    print("    -- new task --")
    tid = _prompt("    id (e.g. t9): ")
    if tid in dag.tasks:
        print(f"    ! id '{tid}' already exists, aborting add.")
        return
    description = _prompt("    description: ")
    class_name = _prompt("    class_name: ") or "Solution"
    function_name = _prompt("    function_name: ")
    params_raw = _prompt("    params, comma-separated (blank = none): ")
    params = [p.strip() for p in params_raw.split(",") if p.strip()]
    deps_raw = _prompt("    depends_on, comma-separated task ids (blank = none): ")
    depends_on = [d.strip() for d in deps_raw.split(",") if d.strip()]
    unknown = [d for d in depends_on if d not in dag.tasks]
    if unknown:
        print(f"    ! unknown task id(s) {unknown}, adding task with no dependencies instead.")
        depends_on = []
    new_task = Task(id=tid, description=description, class_name=class_name,
                     function_name=function_name, params=params, depends_on=depends_on)
    dag.tasks[tid] = new_task
    print(f"    added {tid}.")


def _set_language(dag: TaskDAG):
    print(f"    current target language: {dag.language}")
    new_lang = _prompt("    new language ('python' or 'javascript'): ").strip().lower()
    if new_lang in ("python", "javascript"):
        dag.language = new_lang
        print(f"    set. Future signature() / pretty_print() calls will render {new_lang}-style.")
    else:
        print("    ! only 'python' or 'javascript' supported right now, not applying.")


def _assign_class_ids(dag: TaskDAG) -> dict:
    """Assign a stable id (c1, c2, ...) to each unique class_name, in the
    order those classes first appear in topological order. Needed for
    the memory layer -- classes currently have a name but no id of
    their own, only the tasks (subtasks) under them do."""
    mapping = {}
    for tid in dag.topological_order():
        cname = dag.tasks[tid].class_name
        if cname not in mapping:
            mapping[cname] = f"c{len(mapping) + 1}"
    return mapping


def _revalidate(dag: TaskDAG) -> TaskDAG:
    return TaskDAG(list(dag.tasks.values()), language=dag.language)


def interactive_review(goal: str, dag: TaskDAG, issues: List[Issue]) -> TaskDAG:
    """Walk a human through every detected issue one at a time. Nothing
    is changed unless the human explicitly chooses an action. Returns a
    re-validated TaskDAG. Runs via input() IN THIS SERVICE'S TERMINAL."""
    print(f"\n=== Plan Controller: {len(issues)} issue(s) flagged for review ===")
    if not issues:
        print("(No heuristic issues found -- still recommend a manual skim below.)")

    for i, issue in enumerate(issues, 1):
        print(f"\n--- Issue {i}/{len(issues)} [{issue.kind}] ---")
        print(f"  {issue.message}")
        if issue.detail:
            print(f"  ({issue.detail})")
        for tid in issue.task_ids:
            if tid in dag.tasks:
                _print_task(dag, tid)

        if not issue.task_ids:
            print("  (This issue isn't tied to a specific task -- editing a task "
                  "description won't change it.)")
            while True:
                choice = _prompt(
                    "  Action -- [l]set target language, [n]ew task, "
                    "[c]ontinue to next issue, [q]uit review early: "
                ).lower()
                if choice == "l":
                    _set_language(dag)
                    continue
                if choice == "n":
                    _add_task(dag)
                    continue
                if choice in ("c", "s"):
                    break
                if choice == "q":
                    print("  Ending review early -- remaining issues left as-is.")
                    return _revalidate(dag)
                print("    ! unrecognized choice, try again.")
            continue

        while True:
            choice = _prompt(
                "  Action -- [e]dit description, [d]epends_on, [n]ew task, "
                "[s]kip this issue, [q]uit review early: "
            ).lower()
            if choice == "e":
                tid = _prompt("    which task id? ")
                if tid in dag.tasks:
                    _edit_description(dag, tid)
                else:
                    print("    ! unknown task id.")
                continue
            if choice == "d":
                tid = _prompt("    which task id? ")
                if tid in dag.tasks:
                    _edit_dependencies(dag, tid)
                else:
                    print("    ! unknown task id.")
                continue
            if choice == "n":
                _add_task(dag)
                continue
            if choice == "s":
                break
            if choice == "q":
                print("  Ending review early -- remaining issues left as-is.")
                return _revalidate(dag)
            print("    ! unrecognized choice, try again.")

        while True:
            again = _prompt("  Done with this issue? [Y/n] ").strip().lower()
            if again != "n":
                break
            print(f"\n--- Issue {i}/{len(issues)} [{issue.kind}] (revisiting) ---")
            print(f"  {issue.message}")
            for tid in issue.task_ids:
                if tid in dag.tasks:
                    _print_task(dag, tid)
            while True:
                choice = _prompt(
                    "  Action -- [e]dit description, [d]epends_on, [n]ew task, "
                    "[s]kip this issue, [q]uit review early: "
                ).lower()
                if choice == "e":
                    tid = _prompt("    which task id? ")
                    if tid in dag.tasks:
                        _edit_description(dag, tid)
                    else:
                        print("    ! unknown task id.")
                    continue
                if choice == "d":
                    tid = _prompt("    which task id? ")
                    if tid in dag.tasks:
                        _edit_dependencies(dag, tid)
                    else:
                        print("    ! unknown task id.")
                    continue
                if choice == "n":
                    _add_task(dag)
                    continue
                if choice == "s":
                    break
                if choice == "q":
                    print("  Ending review early -- remaining issues left as-is.")
                    return _revalidate(dag)
                print("    ! unrecognized choice, try again.")

    print("\n--- Free review: add any tasks the heuristics didn't catch ---")
    while _prompt("Add another task? [y/N] ").strip().lower() == "y":
        _add_task(dag)

    return _revalidate(dag)


# ----------------------------------------------------------------------
# FastAPI service
# ----------------------------------------------------------------------
class PlanRequest(BaseModel):
    task_id: str
    goal: str
    backend: str = "mock"
    ollama_model: str = "llama3"
    ollama_url: str = "http://localhost:11434"
    skip_review: bool = False  # True -> detect issues but skip the interactive input() loop


class PlanResponse(BaseModel):
    task_id: str
    status: str                        # "done" | "error"
    goal: Optional[str] = None
    plan: Optional[dict] = None        # TaskDAG.to_dict() shape, enriched with class_id
    issues: Optional[List[dict]] = None
    error: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/plan", response_model=PlanResponse)
def plan_task(req: PlanRequest):
    if req.task_id in _jobs:
        raise HTTPException(
            status_code=409,
            detail=f"task_id '{req.task_id}' was already submitted "
                    "(each task_id must be used once).",
        )

    _jobs[req.task_id] = {
        "status": "running",
        "goal": req.goal,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    cfg = PlannerConfig(model=req.backend, ollama_model=req.ollama_model,
                         ollama_url=req.ollama_url)

    try:
        print(f"\n[plan_controller] starting task_id={req.task_id}: {req.goal[:80]}")
        dag = planner_plan(req.goal, cfg)

        print("\n[plan_controller] plan received back from Planner:")
        print("=== Plan before review ===")
        dag.pretty_print()

        issues = detect_issues(req.goal, dag)

        if req.skip_review:
            print(f"\n[plan_controller] skip_review=True -- returning "
                  f"{len(issues)} issue(s) without interactive review.")
        else:
            dag = interactive_review(req.goal, dag, issues)
            print("\n[plan_controller] plan after human review:")
            print("=== Plan after review ===")
            dag.pretty_print()

    except Exception as e:
        _jobs[req.task_id]["status"] = "error"
        _jobs[req.task_id]["error"] = str(e)
        return PlanResponse(task_id=req.task_id, status="error", error=str(e))

    plan_dict = dag.to_dict()

    class_id_map = _assign_class_ids(dag)
    for t in plan_dict["tasks"]:
        t["class_id"] = class_id_map[t["class_name"]]
    plan_dict["classes"] = [
        {
            "class_id": cid,
            "class_name": cname,
            "task_ids": [t["id"] for t in plan_dict["tasks"] if t["class_name"] == cname],
        }
        for cname, cid in class_id_map.items()
    ]

    result = PlanResponse(
        task_id=req.task_id,
        status="done",
        goal=req.goal,
        plan=plan_dict,
        issues=[i.to_dict() for i in issues],
    )
    _jobs[req.task_id].update({
        "status": "done",
        "goal": req.goal,
        "plan": plan_dict,
        "issues": [i.to_dict() for i in issues],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[plan_controller] finished task_id={req.task_id}: "
          f"{len(plan_dict['tasks'])} subtask(s), {len(issues)} issue(s) flagged")
    return result


@app.get("/jobs/{task_id}")
def get_job(task_id: str):
    if task_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"No task with id '{task_id}'.")
    return _jobs[task_id]
