"""
Coordinator (Step 0 of the multi-agent architecture) — standalone module
=====================================================================
The entry point of the whole pipeline: given a raw goal, decides
whether it goes straight to the Coder or needs the Planner (and, by
extension, Plan Fixer) first.

Routing rule (deterministic, applied in Python -- NOT left to the LLM
to decide directly, so it's auditable and can't silently drift):
  1. If the goal cannot be broken into more than 1 subtask -> coder
  2. Else if the goal is simple                            -> coder
  3. Else                                                    -> planner

The LLM's only job is to ASSESS the goal (estimate subtask count,
judge simplicity, explain why) -- it returns structured JSON, and this
module applies the three rules above to that assessment itself. This
keeps the actual routing logic inspectable and independent of how any
given model phrases its judgment.

DELIBERATELY INDEPENDENT of planner.py / plan_fixer.py / critic.py /
memory.py / coding_agent.py right now, same pattern as those modules:
no imports from them, nothing here assumes how it's called. Wiring it
in (Coordinator decides -> calls planner.py's plan() or hands the goal
straight to coding_agent.py's solve()) is a later step.

Flow (this module only):
  goal (str) -> one LLM call assessing decomposability/simplicity
             -> Decision (route + reasoning), rules applied in Python
"""

import os
import re
import json
from dataclasses import dataclass, asdict
from typing import Optional


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class CoordinatorConfig:
    model: str = "mock"            # "mock" | "ollama" | "openai" | "anthropic"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    request_timeout_sec: int = 60


# ----------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------
@dataclass
class Decision:
    route: str                     # "coder" | "planner"
    rule_applied: str              # which of the 3 rules fired, for auditability
    subtask_estimate: int          # the LLM's estimate of how many subtasks this needs
    is_simple: bool                # the LLM's judgment of rule 2
    reasoning: str                 # short explanation, from the LLM's assessment

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty_print(self):
        print(f"=== Coordinator decision: -> {self.route.upper()} ===")
        print(f"  rule applied: {self.rule_applied}")
        print(f"  subtask estimate: {self.subtask_estimate}")
        print(f"  judged simple: {self.is_simple}")
        print(f"  reasoning: {self.reasoning}")


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------
def _assessment_prompt(goal: str) -> str:
    return f"""You are the routing assessor for a coding pipeline. You do NOT
decide where the task goes -- you only assess two things about it, and
a separate rule-based step uses your assessment to route it.

GOAL:
{goal}

Assess:
1. subtask_estimate: if this goal were broken into independent,
   separately-implementable units of work (e.g. separate
   functions/classes/files that depend on each other), how many would
   there realistically be? A goal answerable with a single function
   is 1. A goal needing several coordinated pieces (e.g. "read data,
   validate it, then write a report") is 2 or more.
2. is_simple: independent of subtask count, is this a simple, well-
   understood, single-concern task (a common utility function, a
   small script, a straightforward transformation) as opposed to
   something with real design complexity, ambiguity, or many
   interacting concerns -- even if it happens to be just 1 subtask?

Respond with ONLY a JSON object, no markdown fences, no prose before
or after, in exactly this shape:

{{
  "subtask_estimate": <integer, 1 or more>,
  "is_simple": true | false,
  "reasoning": "1-2 sentences explaining both judgments"
}}

JSON only -- nothing else."""


# ----------------------------------------------------------------------
# LLM call — same dispatch shape as planner.py/critic.py, duplicated
# on purpose to keep this module standalone.
# ----------------------------------------------------------------------
def call_llm(prompt: str, cfg: CoordinatorConfig) -> str:
    if cfg.model == "mock":
        return _mock_coordinator_llm(prompt)
    if cfg.model == "ollama":
        return call_llm_ollama(prompt, cfg)
    if cfg.model == "openai":
        return call_llm_openai_compatible(prompt)
    if cfg.model == "anthropic":
        return call_llm_anthropic(prompt)
    raise NotImplementedError(f"Unknown backend '{cfg.model}'")


def _mock_coordinator_llm(prompt: str) -> str:
    """Fixed illustrative assessment for the self-contained demo --
    always the same regardless of goal, just like planner.py's mock,
    so the plumbing (parsing/routing/pretty_print) can be tested
    without a real model."""
    return json.dumps({
        "subtask_estimate": 1,
        "is_simple": True,
        "reasoning": "This reads as a single, self-contained function with no "
                      "dependent pieces of work -- a common utility-style task.",
    })


def _extract_json(raw: str) -> dict:
    """Models love wrapping JSON in prose or code fences -- strip both
    and grab the first {...} block. (Same helper as planner.py/critic.py.)"""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if not brace:
        raise RuntimeError(f"Could not find JSON in coordinator output: {raw!r}")
    return json.loads(brace.group(0))


# ----------------------------------------------------------------------
# Routing — the three rules, applied deterministically in Python
# ----------------------------------------------------------------------
def decide(goal: str, cfg: CoordinatorConfig = CoordinatorConfig()) -> Decision:
    raw = call_llm(_assessment_prompt(goal), cfg)
    data = _extract_json(raw)

    subtask_estimate = int(data.get("subtask_estimate", 1))
    if subtask_estimate < 1:
        subtask_estimate = 1
    is_simple = bool(data.get("is_simple", False))
    reasoning = data.get("reasoning", "")

    # Rule 1: can't be broken into more than 1 subtask -> coder
    if subtask_estimate <= 1:
        return Decision(route="coder", rule_applied="1: single subtask, cannot decompose further",
                         subtask_estimate=subtask_estimate, is_simple=is_simple, reasoning=reasoning)

    # Rule 2: task is simple -> coder (even if it has >1 subtask on paper)
    if is_simple:
        return Decision(route="coder", rule_applied="2: judged simple",
                         subtask_estimate=subtask_estimate, is_simple=is_simple, reasoning=reasoning)

    # Rule 3: neither -> planner
    return Decision(route="planner", rule_applied="3: multi-subtask and not simple",
                     subtask_estimate=subtask_estimate, is_simple=is_simple, reasoning=reasoning)


# ----------------------------------------------------------------------
# Real backend hookups (pick one, set CoordinatorConfig(model=...))
# Mirrors planner.py/critic.py's hookups exactly, kept duplicated here
# rather than imported, to keep this module standalone.
# ----------------------------------------------------------------------
def call_llm_ollama(prompt: str, cfg: CoordinatorConfig) -> str:
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
        "max_tokens": 500,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    })
    return resp.json()["content"][0]["text"]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
DEFAULT_GOAL = (
    "Write a function that returns the second largest unique number in a list."
)


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Standalone Coordinator: route a goal to coder or planner")
    p.add_argument("--backend", choices=["mock", "ollama", "openai", "anthropic"],
                    default="mock", help="Which LLM backend to use")
    p.add_argument("--ollama-model", default="llama3")
    p.add_argument("--ollama-url", default="http://localhost:11434")

    p.add_argument("--goal", default=DEFAULT_GOAL, help="The goal to route, given inline")
    p.add_argument("--goal-file", default=None,
                    help="Path to a file containing the goal (overrides --goal)")

    p.add_argument("--output", default=None, help="Write the decision as JSON to this path")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    goal = args.goal
    if args.goal_file:
        with open(args.goal_file, "r", encoding="utf-8") as f:
            goal = f.read().strip()

    cfg = CoordinatorConfig(
        model=args.backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
    )

    try:
        decision = decide(goal, cfg)
    except (RuntimeError, ValueError) as e:
        print(f"\nError: {e}")
        raise SystemExit(1)

    decision.pretty_print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(decision.to_dict(), f, indent=2)
        print(f"\nWrote decision to {args.output}")
