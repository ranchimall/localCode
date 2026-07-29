"""
Critic (Step 5 of the multi-agent architecture) — standalone module
=====================================================================
Takes a finished artifact -- source code, a "solution" (a whole
function/file), or a command-line invocation (ffmpeg, git, curl,
whatever) -- plus whatever context comes with it (what it was
supposed to do, and any error/log output it produced when run) and
asks an LLM to find what's wrong with it.

DELIBERATELY INDEPENDENT of coding_agent.py / planner.py / plan_fixer.py
right now, per the same pattern those modules use: duplicates its own
minimal LLM-calling scaffolding so it can be built and tested in
isolation. Wiring it into the pipeline (as the thing the Tester hands
failures to, whose output the Coder then acts on) is a later step.

Flow (this module only):
  Artifact (content + kind + language + task + error_log)
    -> one LLM call asking for a structured JSON critique
    -> parsed into a CritiqueResult (verdict + summary + Issue list)

Flow (full pipeline, for context -- not implemented here yet):
  Tester runs the artifact -> failure/error output
    -> Critic.critique(artifact_with_error_log)
    -> CritiqueResult handed back to Coder to act on
    -> (later) Memory stores the (artifact, issue, fix) pattern

Swap `call_llm()` for a real backend exactly like planner.py does --
see the three snippets near the bottom of the file (duplicated
on purpose, same reasoning as planner.py: standalone > DRY here).
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
class CriticConfig:
    model: str = "mock"            # "mock" | "ollama" | "openai" | "anthropic"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    request_timeout_sec: int = 60


# ----------------------------------------------------------------------
# What the Critic is asked to look at
# ----------------------------------------------------------------------
@dataclass
class Artifact:
    """The thing being critiqued. `content` is the only required field --
    everything else is context that, if given, the Critic is told to use."""
    content: str                   # the code / command / solution text itself
    kind: str = "code"             # "code" | "command" | "solution" -- free-form, just a hint to the model
    language: str = ""             # "python", "javascript", "bash", "ffmpeg", "html", ... (optional)
    task_description: str = ""     # what it was supposed to do/accomplish (optional)
    error_log: str = ""            # stderr / traceback / test failure output, if it was run (optional)


# ----------------------------------------------------------------------
# Critique result
# ----------------------------------------------------------------------
@dataclass
class Issue:
    kind: str                      # "bug" | "logic" | "syntax" | "security" | "performance" |
                                    # "portability" | "error_mismatch" | "style" | "other"
    severity: str                  # "critical" | "major" | "minor" | "nit"
    message: str                   # human-readable description of the issue
    location: str = ""             # line number / function name / arg position -- whatever the model can point at
    suggested_fix: str = ""        # concrete suggested fix, if the model has one

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "location": self.location,
            "suggested_fix": self.suggested_fix,
        }


_VALID_VERDICTS = {"pass", "needs_fixes", "fail"}
_VALID_SEVERITIES = {"critical", "major", "minor", "nit"}


@dataclass
class CritiqueResult:
    verdict: str                   # "pass" | "needs_fixes" | "fail"
    summary: str                   # 1-3 sentence overall assessment
    issues: List[Issue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "issues": [i.to_dict() for i in self.issues],
        }

    def pretty_print(self):
        order = {"critical": 0, "major": 1, "minor": 2, "nit": 3}
        issues = sorted(self.issues, key=lambda i: order.get(i.severity, 4))

        print(f"=== Critic verdict: {self.verdict.upper()} ===")
        print(f"  {self.summary}")

        if not issues:
            print("\n(No issues found.)")
            return

        print(f"\n=== {len(issues)} issue(s) ===")
        for n, issue in enumerate(issues, 1):
            loc = f"  @ {issue.location}" if issue.location else ""
            print(f"\n  [{n}] [{issue.severity.upper()}] [{issue.kind}]{loc}")
            print(f"      {issue.message}")
            if issue.suggested_fix:
                print(f"      fix -> {issue.suggested_fix}")


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------
def _critic_prompt(artifact: Artifact) -> str:
    kind_label = {"code": "source code", "command": "command-line invocation",
                  "solution": "code solution"}.get(artifact.kind, artifact.kind)

    context_lines = []
    if artifact.language:
        context_lines.append(f"Language/tool: {artifact.language}")
    if artifact.task_description:
        context_lines.append(f"It was supposed to do the following:\n{artifact.task_description}")
    if artifact.error_log:
        context_lines.append(
            "It produced the following error/log output when run "
            f"(cross-check this against the content below -- explain *why* "
            f"this output happened, don't just restate it):\n{artifact.error_log}"
        )
    context = "\n\n".join(context_lines)

    return f"""You are a careful, skeptical code/command reviewer (a "Critic" step
in an automated coding pipeline). You will be shown a {kind_label} and
asked to find real, concrete problems with it -- not style nitpicks
dressed up as bugs, and not invented issues to seem thorough.

{context}

--- CONTENT TO REVIEW ---
{artifact.content}
--- END CONTENT ---

Check for things like (as applicable to what you're shown):
- Syntax errors or invalid usage (invalid flags/arguments for a CLI tool, etc.)
- Logic bugs: does it actually do what the task description says?
- Edge cases / inputs likely to break it
- If an error log was given: does the content actually explain that
  error? Point to the specific line/argument responsible.
- Security issues (injection, unsafe eval, secrets, unsafe permissions)
- Portability issues (OS-specific assumptions, hardcoded paths)
- Only flag style/performance if nothing more serious is wrong, and
  mark those "minor" or "nit" -- don't let them crowd out real bugs

Respond with ONLY a JSON object, no markdown fences, no prose before
or after, in exactly this shape:

{{
  "verdict": "pass" | "needs_fixes" | "fail",
  "summary": "1-3 sentence overall assessment",
  "issues": [
    {{
      "kind": "bug" | "logic" | "syntax" | "security" | "performance" | "portability" | "error_mismatch" | "style" | "other",
      "severity": "critical" | "major" | "minor" | "nit",
      "message": "what's wrong, specifically",
      "location": "line number / function name / arg -- whatever you can point to, or empty string",
      "suggested_fix": "a concrete fix, or empty string if you don't have one"
    }}
  ]
}}

Rules:
- "pass": no real issues, or only "nit"-level ones.
- "fail": at least one "critical" issue (it won't run / does the wrong
  thing / actively unsafe).
- "needs_fixes": anything in between.
- If you have no issues, return "issues": [].
- JSON only -- nothing else."""


# ----------------------------------------------------------------------
# LLM call — same dispatch shape as planner.py, duplicated on purpose
# ----------------------------------------------------------------------
def call_llm(prompt: str, cfg: CriticConfig) -> str:
    if cfg.model == "mock":
        return _mock_critic_llm(prompt)
    if cfg.model == "ollama":
        return call_llm_ollama(prompt, cfg)
    if cfg.model == "openai":
        return call_llm_openai_compatible(prompt)
    if cfg.model == "anthropic":
        return call_llm_anthropic(prompt)
    raise NotImplementedError(f"Unknown backend '{cfg.model}'")


def _mock_critic_llm(prompt: str) -> str:
    """Fixed illustrative critique for the self-contained demo -- always
    the same regardless of input, just like planner.py's mock, so the
    plumbing (parsing/pretty_print) can be tested without a real model."""
    return json.dumps({
        "verdict": "needs_fixes",
        "summary": "The core logic looks reasonable but there's an unhandled "
                    "edge case and the error log points to a real bug.",
        "issues": [
            {
                "kind": "bug",
                "severity": "major",
                "message": "Division by zero isn't guarded against when the "
                            "input collection is empty.",
                "location": "line 12, function `average`",
                "suggested_fix": "Add a check: return 0 (or raise a clear "
                                  "error) if the collection is empty before dividing.",
            },
            {
                "kind": "style",
                "severity": "nit",
                "message": "Variable name `x` is not descriptive.",
                "location": "line 4",
                "suggested_fix": "Rename to something like `total`.",
            },
        ],
    })


def _extract_json(raw: str) -> dict:
    """Models love wrapping JSON in prose or code fences -- strip both
    and grab the first {...} block. (Same helper as planner.py.)"""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if not brace:
        raise RuntimeError(f"Could not find JSON in critic output: {raw!r}")
    return json.loads(brace.group(0))


def critique(artifact: Artifact, cfg: CriticConfig = CriticConfig()) -> CritiqueResult:
    """The main entry point: Artifact -> CritiqueResult."""
    raw = call_llm(_critic_prompt(artifact), cfg)
    data = _extract_json(raw)

    verdict = data.get("verdict", "needs_fixes")
    if verdict not in _VALID_VERDICTS:
        verdict = "needs_fixes"

    issues = []
    for i in data.get("issues", []):
        severity = i.get("severity", "minor")
        if severity not in _VALID_SEVERITIES:
            severity = "minor"
        issues.append(Issue(
            kind=i.get("kind", "other"),
            severity=severity,
            message=i.get("message", ""),
            location=i.get("location", ""),
            suggested_fix=i.get("suggested_fix", ""),
        ))

    return CritiqueResult(
        verdict=verdict,
        summary=data.get("summary", ""),
        issues=issues,
    )


# ----------------------------------------------------------------------
# Real backend hookups (pick one, set CriticConfig(model=...))
# Mirrors planner.py's hookups exactly, kept duplicated here rather
# than imported, to keep this module standalone.
# ----------------------------------------------------------------------
def call_llm_ollama(prompt: str, cfg: CriticConfig) -> str:
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
DEFAULT_CONTENT = (
    "def average(nums):\n"
    "    x = sum(nums)\n"
    "    return x / len(nums)\n"
)


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Standalone Critic: review code/commands/solutions for issues")
    p.add_argument("--backend", choices=["mock", "ollama", "openai", "anthropic"],
                    default="mock", help="Which LLM backend to use")
    p.add_argument("--ollama-model", default="llama3")
    p.add_argument("--ollama-url", default="http://localhost:11434")

    p.add_argument("--code", default=None, help="The code/command/solution text to review, given inline")
    p.add_argument("--code-file", default=None,
                    help="Path to a file containing the code/command/solution to review (overrides --code)")

    p.add_argument("--kind", choices=["code", "command", "solution"], default="code",
                    help="What kind of thing is being reviewed")
    p.add_argument("--language", default="", help="e.g. python, javascript, bash, ffmpeg, html")

    p.add_argument("--task", default="", help="What the content was supposed to do")
    p.add_argument("--task-file", default=None, help="Path to a file with the task description (overrides --task)")

    p.add_argument("--error-log", default="", help="Error/traceback/test-failure output, if it was run")
    p.add_argument("--error-log-file", default=None,
                    help="Path to a file with the error/log output (overrides --error-log)")

    p.add_argument("--output", default=None, help="Write the critique as JSON to this path")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    content = args.code or DEFAULT_CONTENT
    if args.code_file:
        with open(args.code_file, "r", encoding="utf-8") as f:
            content = f.read()

    task_description = args.task
    if args.task_file:
        with open(args.task_file, "r", encoding="utf-8") as f:
            task_description = f.read().strip()

    error_log = args.error_log
    if args.error_log_file:
        with open(args.error_log_file, "r", encoding="utf-8") as f:
            error_log = f.read()

    artifact = Artifact(
        content=content,
        kind=args.kind,
        language=args.language,
        task_description=task_description,
        error_log=error_log,
    )

    cfg = CriticConfig(
        model=args.backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
    )

    try:
        result = critique(artifact, cfg)
    except (RuntimeError, ValueError) as e:
        print(f"\nError: {e}")
        raise SystemExit(1)

    result.pretty_print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nWrote critique to {args.output}")
