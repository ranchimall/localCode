"""
Planner (Step 2 of the multi-agent architecture) — standalone module
======================================================================
Takes a single high-level coding goal and produces a small, ordered
DAG of subtasks — each already shaped into a function name, an owning
class (OOP grouping), parameters, and dependencies on other subtasks —
ready for a future Worker step to actually implement.

DELIBERATELY INDEPENDENT of coding_agent.py right now, per instruction.
It duplicates the minimal LLM-calling scaffolding it needs rather than
importing anything, so it can be built and tested in isolation. Wiring
it into the main pipeline (as the thing that hands tasks to the
existing generate-tests/generate-code/verify loop) is a later step.

Flow:
  goal (str)
    -> one LLM call asking for a structured JSON breakdown
    -> parsed into Task objects
    -> assembled into a TaskDAG (validates: no missing deps, no cycles)
    -> TaskDAG.topological_order() / .classes() for downstream use

Swap `call_llm()` for a real backend exactly like coding_agent.py does
— see the three snippets near the bottom of the file.
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class PlannerConfig:
    model: str = "mock"            # "mock" | "ollama" | "openai" | "anthropic"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    request_timeout_sec: int = 60


# ----------------------------------------------------------------------
# Task object
# ----------------------------------------------------------------------
@dataclass
class Task:
    id: str
    description: str                       # plain-English subtask
    class_name: str                        # owning class (OOP grouping)
    function_name: str                     # method implementing it
    params: List[str] = field(default_factory=list)
    returns: str = ""
    depends_on: List[str] = field(default_factory=list)

    def signature(self, language: str = "python") -> str:
        if language == "javascript":
            return f"function {self.function_name}({', '.join(self.params)}) {{"
        params = ", ".join(["self"] + self.params) if self.params else "self"
        return f"def {self.function_name}({params}):"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "class_name": self.class_name,
            "function_name": self.function_name,
            "params": self.params,
            "returns": self.returns,
            "depends_on": self.depends_on,
        }


# ----------------------------------------------------------------------
# TaskDAG — lightweight, dependency-free (no networkx): just enough
# for cycle detection + topological ordering + class grouping. Swap
# for a real graph lib later if the Orchestrator needs more.
# ----------------------------------------------------------------------
class TaskDAG:
    def __init__(self, tasks: List[Task], language: str = "python"):
        self.tasks = {t.id: t for t in tasks}
        self.language = language
        if len(self.tasks) != len(tasks):
            raise ValueError("Duplicate task ids in planner output.")
        self._validate()

    def _validate(self):
        for t in self.tasks.values():
            for dep in t.depends_on:
                if dep not in self.tasks:
                    raise ValueError(
                        f"Task '{t.id}' depends on unknown task '{dep}'"
                    )
        self.topological_order()  # raises ValueError on cycles as a side effect

    def topological_order(self) -> List[str]:
        visited, visiting, order = set(), set(), []

        def visit(tid):
            if tid in visited:
                return
            if tid in visiting:
                raise ValueError(f"Cycle detected in task DAG at '{tid}'")
            visiting.add(tid)
            for dep in self.tasks[tid].depends_on:
                visit(dep)
            visiting.discard(tid)
            visited.add(tid)
            order.append(tid)

        for tid in self.tasks:
            visit(tid)
        return order

    def to_dict(self) -> dict:
        """Serialize back into the same {"tasks": [...]} shape the planner
        LLM produces, so it can be dumped to JSON, hand-edited by a human,
        or reloaded by PlanFixer."""
        return {"language": self.language,
                "tasks": [self.tasks[tid].to_dict() for tid in self.topological_order()]}

    @classmethod
    def from_dict(cls, data: dict) -> "TaskDAG":
        tasks = []
        for t in data.get("tasks", []):
            tasks.append(Task(
                id=t["id"],
                description=t.get("description", ""),
                class_name=t.get("class_name", "Solution"),
                function_name=t["function_name"],
                params=t.get("params", []),
                returns=t.get("returns", ""),
                depends_on=t.get("depends_on", []),
            ))
        if not tasks:
            raise RuntimeError("Plan has zero tasks.")
        return cls(tasks, language=data.get("language", "python"))

    def classes(self) -> dict:
        """Group tasks by owning class_name -> list[Task], giving the
        OOP structure the planner was asked to produce."""
        grouped = {}
        for tid in self.topological_order():
            t = self.tasks[tid]
            grouped.setdefault(t.class_name, []).append(t)
        return grouped

    def pretty_print(self):
        order = self.topological_order()
        print("=== TASK DAG (topological order) ===")
        for tid in order:
            t = self.tasks[tid]
            deps = f"  (depends on: {', '.join(t.depends_on)})" if t.depends_on else ""
            print(f"  [{t.id}] {t.class_name}.{t.function_name}"
                  f"({', '.join(t.params)}){deps}")
            print(f"      \"{t.description}\"")

        print("\n=== CLASSES (OOP grouping) ===")
        for class_name, tasks in self.classes().items():
            print(f"\nclass {class_name}:")
            for t in tasks:
                print(f"    {t.signature(self.language)}  # {t.description}")


# ----------------------------------------------------------------------
# LLM call — replace with a real backend for actual use
# ----------------------------------------------------------------------
def call_llm(prompt: str, cfg: PlannerConfig) -> str:
    if cfg.model == "mock":
        return _mock_planner_llm(prompt)
    if cfg.model == "ollama":
        return call_llm_ollama(prompt, cfg)
    if cfg.model == "openai":
        return call_llm_openai_compatible(prompt)
    if cfg.model == "anthropic":
        return call_llm_anthropic(prompt)
    raise NotImplementedError(f"Unknown backend '{cfg.model}'")


def _mock_planner_llm(prompt: str) -> str:
    """Fixed illustrative breakdown for the self-contained demo —
    always the same regardless of goal text, just like coding_agent.py's
    mock, so the plumbing (parsing/DAG/class-grouping) can be tested
    without a real model."""
    return json.dumps({
        "goal_actions": ["read files", "extract emails", "deduplicate emails", "save to file"],
        "tasks": [
            {"id": "t1",
             "description": "Read all .txt files from the given folder",
             "class_name": "FileReader", "function_name": "read_txt_files",
             "params": ["folder_path"], "returns": "list of file contents (str)",
             "depends_on": []},
            {"id": "t2",
             "description": "Extract email addresses from text using regex",
             "class_name": "EmailExtractor", "function_name": "extract_emails",
             "params": ["text"], "returns": "list of email strings",
             "depends_on": ["t1"]},
            {"id": "t3",
             "description": "Remove duplicate email addresses, preserving order",
             "class_name": "Deduplicator", "function_name": "dedupe",
             "params": ["emails"], "returns": "list of unique email strings",
             "depends_on": ["t2"]},
            {"id": "t4",
             "description": "Save the final list of emails to a new file",
             "class_name": "FileWriter", "function_name": "save_emails",
             "params": ["emails", "output_path"], "returns": "None",
             "depends_on": ["t3"]},
        ]
    })


# ----------------------------------------------------------------------
# Prompt + parsing
# ----------------------------------------------------------------------
def _planning_prompt(goal: str) -> str:
    return f"""You are a software planning assistant. Break the following
coding goal into a small ordered set of subtasks, then shape each
subtask into a function that belongs to a class (object-oriented
design), and record dependencies between subtasks as a DAG.

GOAL:
{goal}

IMPORTANT -- before writing tasks, first identify every distinct
action the goal asks for (e.g. "read", "extract", "deduplicate",
"save"). Every one of those actions MUST show up, explicitly named,
in at least one task's "description". Do not silently fold two
distinct actions into one function without mentioning both actions
in that function's description -- if a function does two things,
say so, or split it into two tasks instead.

Respond with ONLY a JSON object (no prose, no markdown fences) in
exactly this shape:

{{
  "goal_actions": ["short phrase for each distinct action in the goal"],
  "tasks": [
    {{
      "id": "t1",
      "description": "short plain-English description of this subtask",
      "class_name": "PascalCaseClassName",
      "function_name": "snake_case_function_name",
      "params": ["param1", "param2"],
      "returns": "short description of the return value/type",
      "depends_on": []
    }}
  ]
}}

Rules:
- 3-7 tasks, each doing ONE clear thing
- depends_on lists the "id"s of tasks that must run first (use [] if none)
- Group related functions under the same class_name where it makes sense
- No task should depend on itself or create a cycle
- Every entry in "goal_actions" must be reflected in at least one task description
- JSON only -- nothing else, no markdown fences"""


def _extract_json(raw: str) -> dict:
    """Models love wrapping JSON in prose or code fences -- strip both
    and grab the first {...} block. (Same helper as coordinator.py/
    critic.py.) Also tolerates the single most common way real models
    break strict JSON: a trailing comma before a closing ] or } (valid
    in JS-style object literals, invalid in JSON) -- e.g.
    {"depends_on": ["t2"],} would otherwise raise
    'Expecting property name enclosed in double quotes'."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if not brace:
        raise RuntimeError(f"Could not find JSON in planner output: {raw!r}")
    candidate = brace.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # strip trailing commas before a closing bracket/brace and retry once
        cleaned = re.sub(r",(\s*[\]}])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Planner output wasn't valid JSON even after cleanup "
                f"({e}). Raw model output:\n{raw}"
            )


def _check_action_coverage(goal_actions: list, tasks: list) -> list:
    """Heuristic safety net: for each action the model itself said the
    goal requires, confirm at least one task's description/function
    name/class name plausibly covers it. Returns the list of actions
    that don't seem covered by anything -- an early warning that a
    requirement silently got dropped between the goal and the plan,
    the same failure mode as the naming and behavior-rule mismatches
    earlier in this project. This doesn't fix the plan automatically
    -- it just makes a silent drop visible instead of invisible.

    Uses substring-based stem matching (not exact word match) since
    word forms legitimately vary -- e.g. a goal action "deduplicate"
    should be recognized as covered by a task that says "duplicate"
    or "dedupe", not just an exact literal match."""
    haystack_words = set()
    for t in tasks:
        text = f"{t.description} {t.function_name} {t.class_name}".lower()
        haystack_words.update(re.findall(r"[a-z]+", text))

    def word_covered(word: str) -> bool:
        if word in haystack_words:
            return True
        for hw in haystack_words:
            if min(len(word), len(hw)) < 4:
                continue
            if word in hw or hw in word:
                return True
            # shared-prefix fallback: catches same-root/different-suffix
            # pairs that pure substring matching misses, e.g.
            # "validate"/"validation", "convert"/"conversion" -- these
            # diverge mid-word so neither is a substring of the other,
            # but share a long enough root to be the same concept.
            prefix_len = min(5, len(word), len(hw))
            if word[:prefix_len] == hw[:prefix_len]:
                return True
        return False

    missing = []
    for action in goal_actions:
        words = [w for w in re.findall(r"[a-z]+", action.lower()) if len(w) > 2]
        if words and not all(word_covered(w) for w in words):
            missing.append(action)
    return missing


def plan(goal: str, cfg: PlannerConfig = PlannerConfig()) -> TaskDAG:
    """The main entry point: goal -> validated TaskDAG."""
    raw = call_llm(_planning_prompt(goal), cfg)
    data = _extract_json(raw)

    if not data.get("tasks"):
        raise RuntimeError("Planner returned zero tasks.")

    dag = TaskDAG.from_dict(data)

    goal_actions = data.get("goal_actions", [])
    missing = _check_action_coverage(goal_actions, list(dag.tasks.values()))
    if missing:
        print(f"[planner] WARNING: these actions from the goal don't "
              f"appear to be covered by any task -- they may have been "
              f"silently dropped: {missing}")

    return dag


# ----------------------------------------------------------------------
# Real backend hookups (pick one, set PlannerConfig(model=...))
# Mirrors coding_agent.py's hookups exactly, kept duplicated here
# rather than imported, to keep this module standalone.
# ----------------------------------------------------------------------
def call_llm_ollama(prompt: str, cfg: PlannerConfig) -> str:
    import requests
    try:
        resp = requests.post(
            f"{cfg.ollama_url}/api/generate",
            json={
                "model": cfg.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=cfg.request_timeout_sec,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Couldn't reach Ollama at {cfg.ollama_url}. "
            f"Is it running? Start it with `ollama serve`."
        )

    if resp.status_code == 404:
        raise RuntimeError(
            f"Model '{cfg.ollama_model}' isn't pulled. "
            f"Run `ollama pull {cfg.ollama_model}` first, "
            f"or pass --ollama-model with one from `ollama list`."
        )
    resp.raise_for_status()
    return resp.json()["response"]


def call_llm_openai_compatible(prompt: str) -> str:
    import requests
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "your-model",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    })
    return resp.json()["choices"][0]["message"]["content"]


def call_llm_anthropic(prompt: str) -> str:
    import requests
    resp = requests.post("https://api.anthropic.com/v1/messages", headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }, json={
        "model": "claude-sonnet-4-6",
        "max_tokens": 1500,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    })
    return resp.json()["content"][0]["text"]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
DEFAULT_GOAL = (
    "Build a script that reads a folder of .txt files, extracts all "
    "email addresses using regex, deduplicates them, and saves to a "
    "new file"
)


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Standalone task planner")
    p.add_argument("--backend", choices=["mock", "ollama", "openai", "anthropic"],
                    default="mock", help="Which LLM backend to use")
    p.add_argument("--ollama-model", default="llama3")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--goal", default=DEFAULT_GOAL, help="High-level coding goal")
    p.add_argument("--goal-file", default=None,
                    help="Path to a text file containing the goal "
                         "(overrides --goal)")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    goal = args.goal
    if args.goal_file:
        with open(args.goal_file, "r", encoding="utf-8") as f:
            goal = f.read().strip()

    cfg = PlannerConfig(
        model=args.backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
    )

    try:
        dag = plan(goal, cfg)
    except (RuntimeError, ValueError) as e:
        print(f"\nError: {e}")
        raise SystemExit(1)

    dag.pretty_print()
