"""
Self-Correcting Coding Agent (with Best-of-N candidate generation)
====================================================================
Generates code + tests, runs them in a lightly-sandboxed subprocess,
and iterates on failures — tracking the best-scoring attempt rather
than just the last one.

NEW IN THIS VERSION: at each round the agent generates N candidates
in parallel (initial solutions on round 0, fix-attempts on later
rounds) instead of a single one, scores all of them, and keeps the
best. This trades API cost for fewer iterations / higher hit-rate,
and is the same idea behind "best-of-N sampling" in serious agent
systems.

Swap `call_llm()` for a real backend (Ollama / OpenAI-compatible /
Anthropic) — see the three snippets near the bottom of the file.
"""

import os
import re
import sys
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class AgentConfig:
    max_iters: int = 20
    timeout_sec: int = 5
    cpu_seconds: int = 2          # RLIMIT_CPU inside the sandboxed process
    memory_mb: int = 256          # RLIMIT_AS inside the sandboxed process
    model: str = "mock"           # "mock" | "ollama" | "openai" | "anthropic"
    n_candidates: int = 8         # best-of-N width per round
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"  # swap for any locally-pulled model, e.g. "qwen2.5-coder", "codellama", "deepseek-coder-v2"
    request_timeout_sec: int = 60
    verify_tests: bool = False    # human-in-the-loop test verification (see human_verify_tests)
    auto_verify_after_rounds: int = 5
    # If still failing after this many rounds (and verify_tests wasn't already
    # on), pause and auto-trigger human_verify_tests -- persistent failure
    # across many rounds/candidates is usually a sign the generated TESTS are
    # wrong (contradictory, or asserting the wrong thing), not that the code
    # is wrong. Set to 0 to disable.


# ----------------------------------------------------------------------
# LLM call — replace with a real backend for actual use
# ----------------------------------------------------------------------
def call_llm(prompt: str, cfg: AgentConfig) -> str:
    if cfg.model == "mock":
        return _mock_llm(prompt)
    if cfg.model == "ollama":
        return call_llm_ollama(prompt, cfg)
    if cfg.model == "openai":
        return call_llm_openai_compatible(prompt)
    if cfg.model == "anthropic":
        return call_llm_anthropic(prompt)
    raise NotImplementedError(f"Unknown backend '{cfg.model}'")


_CANDIDATE_MARKER = re.compile(r"\[\[candidate=(\d+)\]\]")

_SIGNATURE_RE = re.compile(r'def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?:')


def _parse_signature(raw: str) -> dict:
    """Extract {'name', 'params', 'line'} from the model's raw
    signature-gen output, tolerating stray prose/code-fences around
    the one line we actually need."""
    cleaned = strip_code_fence(raw)
    m = _SIGNATURE_RE.search(cleaned)
    if not m:
        raise RuntimeError(f"Could not parse a function signature from: {raw!r}")
    name, params = m.group(1), m.group(2).strip()
    return {"name": name, "params": params, "line": f"def {name}({params}):"}


def _tag_candidate(prompt: str, candidate_id: int) -> str:
    """Embed a candidate id so a real LLM call can be given a distinct
    seed/temperature per call, and so the mock backend can vary its
    output to simulate sampling diversity."""
    return f"[[candidate={candidate_id}]]\n{prompt}"


def _extract_candidate_id(prompt: str):
    m = _CANDIDATE_MARKER.search(prompt)
    return int(m.group(1)) if m else None


def _mock_llm(prompt: str) -> str:
    """
    Fake backend for the self-contained demo. Branches on prompt type
    AND candidate id, so different candidates in a best-of-N round
    actually differ — some buggy, some correct — the way real sampled
    completions would.
    """
    cid = _extract_candidate_id(prompt) or 0

    if "respond with ONLY the exact function" in prompt:
        return "def second_largest_unique(nums):"

    if "Write Python tests" in prompt:
        return '''
def test_case_1():
    assert second_largest_unique([1, 2, 3, 4]) == 3

def test_case_2():
    assert second_largest_unique([5, 5, 5, 5]) is None

def test_case_3():
    assert second_largest_unique([10, 20]) == 10

def test_case_4():
    assert second_largest_unique([]) is None

def test_case_5():
    assert second_largest_unique([7]) is None
'''

    if "Fix the code" in prompt:
        # Two of every four "fix" candidates land on the correct fix,
        # the others land on a still-slightly-wrong variant, to show
        # best-of-N helping the *repair* step too, not just round 0.
        if cid % 2 == 0:
            return '''
def second_largest_unique(nums):
    uniq = sorted(set(nums), reverse=True)
    if len(uniq) < 2:
        return None
    return uniq[1]
'''
        else:
            return '''
def second_largest_unique(nums):
    uniq = sorted(set(nums), reverse=True)
    return uniq[1]  # still crashes on <2 unique values
'''

    # Code-gen round: 4 distinct candidates, 2 buggy / 2 correct,
    # mimicking real sampling variance.
    variants = [
        # 0: buggy — doesn't dedupe
        '''
def second_largest_unique(nums):
    s = sorted(nums, reverse=True)
    if len(s) < 2:
        return None
    return s[1]
''',
        # 1: correct
        '''
def second_largest_unique(nums):
    uniq = sorted(set(nums), reverse=True)
    if len(uniq) < 2:
        return None
    return uniq[1]
''',
        # 2: buggy — wrong index after dedupe
        '''
def second_largest_unique(nums):
    uniq = sorted(set(nums), reverse=True)
    if len(uniq) < 2:
        return None
    return uniq[0]
''',
        # 3: correct — alternate implementation
        '''
def second_largest_unique(nums):
    uniq = set(nums)
    if len(uniq) < 2:
        return None
    uniq.discard(max(uniq))
    return max(uniq)
''',
    ]
    return variants[cid % len(variants)]


def strip_code_fence(text: str) -> str:
    """Remove ```python ... ``` / ``` ... ``` wrappers models love to add."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ----------------------------------------------------------------------
# Sandboxed execution
# ----------------------------------------------------------------------
def _limit_resources(cfg: AgentConfig):
    """preexec_fn: applied inside the child process before exec.
    POSIX-only (the `resource` module doesn't exist on Windows at all,
    not even as a stub) — callers must guard with os.name == "posix"
    before passing this as preexec_fn, since preexec_fn itself is also
    unsupported on Windows subprocess calls."""
    import resource

    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cfg.cpu_seconds, cfg.cpu_seconds))
        mem_bytes = cfg.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    return _apply


def _sandbox_env() -> dict:
    """
    A scrubbed environment for the child process — strips inherited
    proxy settings, API keys, etc. that LLM-generated code has no
    business seeing. But it can't be *too* scrubbed: on Windows,
    Python/pytest need several OS-level vars just to launch at all
    (missing SystemRoot in particular can break socket/runtime init
    before your code ever runs). Keep only what's needed to boot.
    """
    if os.name == "nt":
        keep = ["PATH", "SystemRoot", "TEMP", "TMP", "USERPROFILE",
                "PATHEXT", "COMSPEC", "windir", "APPDATA", "LOCALAPPDATA",
                "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"]
    else:
        keep = ["PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"]
    return {k: os.environ[k] for k in keep if k in os.environ}


def run_code(code: str, tests: str, cfg: AgentConfig) -> dict:
    """
    Writes solution + tests to a temp dir and runs them under pytest
    with JUnit XML output, giving a per-test-case pass/fail count.

    NOTE ON SANDBOXING: RLIMIT_CPU / RLIMIT_AS + a scrubbed env stop
    runaway loops and most memory bombs (POSIX only, no-op on Windows),
    but do NOT stop filesystem or network access. For genuinely
    untrusted code, wrap this in a container with no network and a
    disposable filesystem instead of relying on process-level limits.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        solution_path = os.path.join(tmpdir, "solution.py")
        test_path = os.path.join(tmpdir, "test_solution.py")
        report_path = os.path.join(tmpdir, "report.xml")

        with open(solution_path, "w") as f:
            f.write(strip_code_fence(code))

        with open(test_path, "w") as f:
            f.write("from solution import *\n\n" + strip_code_fence(tests))

        env = _sandbox_env()

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", test_path,
                 f"--junitxml={report_path}", "-q"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=cfg.timeout_sec,
                env=env,
                preexec_fn=_limit_resources(cfg) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "score": 0.0, "total": 0, "failed_count": 0,
                    "stderr": f"Timed out after {cfg.timeout_sec}s (possible infinite loop)"}
        except Exception as e:
            return {"passed": False, "score": 0.0, "total": 0, "failed_count": 0,
                    "stderr": f"Execution error: {e}"}

        if not os.path.exists(report_path):
            return {"passed": False, "score": 0.0, "total": 0, "failed_count": 0,
                    "stderr": proc.stderr or proc.stdout}

        return _parse_junit(report_path, proc.stderr)


def _parse_junit(report_path: str, stderr: str) -> dict:
    tree = ET.parse(report_path)
    root = tree.getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")

    total = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    passed = total - failures
    score = (passed / total) if total else 0.0

    failure_msgs = []
    for case in suite.iter("testcase"):
        for tag in ("failure", "error"):
            node = case.find(tag)
            if node is not None:
                detail = (node.text or "").strip()
                summary = detail if detail else node.get("message", "")
                failure_msgs.append(f"{case.get('name')}: {summary}")

    return {
        "passed": failures == 0 and total > 0,
        "score": score,
        "total": total,
        "failed_count": failures,
        "stderr": "\n".join(failure_msgs) or stderr,
    }



# ----------------------------------------------------------------------
# Prompt templates
# ----------------------------------------------------------------------

def _signature_gen_prompt(task: str) -> str:
    return f"""Given this coding task, respond with ONLY the exact function
signature that should be implemented — nothing else.

TASK:
{task}

Rules:
- One line, in the form: def function_name(param1, param2, ...):
- No body, no docstring, no explanation, no markdown fences
- Pick a clear, conventional Python name based on the task"""


def _code_gen_prompt(task: str, signature: dict) -> str:
    return f"""Write Python code for this task:

{task}
Your solution MUST define a function with exactly this signature:
{signature['line']}

Rules:
- Only return code
- No explanations"""


def _test_gen_prompt(task: str, signature: dict) -> str:
    return f"""Write Python tests for this task:

{task}

The function under test MUST match this exact signature:
{signature['line']}

Rules:
- Write separate functions named test_case_1, test_case_2, ... (do NOT bundle into one test_all)
- Call the function using exactly the name `{signature['name']}`
- Use assert statements
- Cover edge cases
- Do NOT define, implement, or stub out `{signature['name']}` yourself -- ONLY
  call it. It will be imported from a separately-generated solution file; a
  `def {signature['name']}(...)` written here will silently override that
  import and break every test.
- All tests must be consistent with exactly one correct implementation. If
  `{signature['name']}` takes no arguments (or the same arguments across
  tests), every test calling it must expect the SAME return value -- do not
  write two tests with different expected outputs for the same inputs.
- No explanations, no markdown fences, no code outside the test_case_N functions"""


def _fix_prompt(task: str, code: str, tests: str, error: str, signature: dict) -> str:
    return f"""The following code is failing tests.

TASK:
{task}

REQUIRED SIGNATURE (do not rename or change the parameter list):
{signature['line']}

CODE:
{code}

TESTS:
{tests}

ERROR:
{error}

Fix the code.

Rules:
- Only return corrected code
- Keep the function named exactly `{signature['name']}`
- Do not modify tests"""


def _strip_foreign_defs(tests: str, signature: dict) -> str:
    """Defensive guard, independent of prompt compliance: the test-gen
    LLM sometimes ignores instructions and writes a stub/implementation
    of the solution function inside the tests text. Since the test file
    is built as `from solution import *` followed by the generated
    tests, any later `def {signature['name']}` here silently shadows
    the real candidate implementation -- every test then calls the
    stub, not the code under test, and scores become meaningless
    (this is exactly what happened when a `def find_number(): pass`
    stub leaked into a generated test file and made every candidate
    fail with `None == 5` regardless of correctness). Strip any
    top-level def block matching the solution's name before it's ever
    written to disk."""
    name = re.escape(signature["name"])
    pattern = re.compile(
        rf"^def\s+{name}\s*\([^)]*\)\s*(?:->[^:]+)?:\n(?:[ \t].*\n?|\n)*",
        re.MULTILINE,
    )
    cleaned, n = pattern.subn("", tests)
    if n:
        print(f"[coding_agent] WARNING: stripped {n} stray definition(s) of "
              f"'{signature['name']}' from generated tests -- the test-gen "
              f"model included an implementation/stub despite instructions "
              f"not to. This would otherwise shadow the real solution and "
              f"make every candidate fail regardless of correctness.")
    return cleaned


def generate_signature(task: str, cfg: AgentConfig) -> dict:
    raw = call_llm(_signature_gen_prompt(task), cfg)
    return _parse_signature(raw)


def generate_tests(task: str, cfg: AgentConfig, signature: dict) -> str:
    return call_llm(_test_gen_prompt(task, signature), cfg)

# ----------------------------------------------------------------------
# Human-in-the-loop test verification
# ----------------------------------------------------------------------
# This only handles the simple, single-assert-per-function style our
# own _test_gen_prompt asks for: `assert func(args) == expected` (or
# `is expected`), one per `def test_case_N():`. Anything that doesn't
# match that shape is left untouched and passed through unverified,
# with a note printed so nothing silently vanishes.
_TEST_DEF_RE = re.compile(r'\s*def\s+(test_case_\d+)\s*\(')
_ASSERT_RE = re.compile(
    r'\s*assert\s+(?P<func>\w+)\((?P<args>.*)\)\s*(?P<op>==|is)\s*'
    r'(?P<expected>[^,]+?)\s*(?:,\s*(?P<msg>.+))?\s*$'
)


def _parse_test_cases(tests: str) -> list:
    """Extract (name, func, args, op, expected, msg) for every test
    that follows the one-assert-per-function shape."""
    cases = []
    current_name = None
    for line in tests.splitlines():
        m_def = _TEST_DEF_RE.match(line)
        if m_def:
            current_name = m_def.group(1)
            continue
        if current_name is None:
            continue
        m_assert = _ASSERT_RE.match(line)
        if m_assert:
            d = m_assert.groupdict()
            cases.append({
                "name": current_name, "func": d["func"], "args": d["args"].strip(),
                "op": d["op"], "expected_raw": d["expected"].strip(), "msg": d["msg"],
            })
            current_name = None  # one assert per test_case_N
    return cases


def human_verify_tests(tests: str, task: str) -> str:
    """Show each generated test's input -> expected output to a human
    and let them confirm, correct, or drop it — before any solution
    code is scored against it."""
    cases = _parse_test_cases(tests)
    if not cases:
        print("[verify] Couldn't parse individual test cases from this output "
              "(unexpected format) — skipping verification, using generated "
              "tests as-is.")
        return tests

    print(f"\n=== Verify {len(cases)} generated test case(s) yourself ===")
    print(f"Task: {task}")
    print("You don't need to know the solution — just whether the expected "
          "result shown is actually correct.\n")

    approved = []
    for i, case in enumerate(cases, 1):
        print(f"Test {i}/{len(cases)}  [{case['name']}]:  {case['args']}  ->  {case['expected_raw']}")
        while True:
            resp = input("  Correct? [Enter/y]es, [n]o - fix it, [s]kip this test: ").strip().lower()
            if resp in ("", "y", "yes"):
                approved.append(case)
                break
            if resp in ("n", "no"):
                fixed = input("  What should the expected result actually be? "
                               "(e.g. True / False / None / 42): ").strip()
                if fixed:
                    case = {**case, "expected_raw": fixed}
                approved.append(case)
                break
            if resp in ("s", "skip"):
                print("  Skipped - won't be used to score any solution.")
                break
            print("  Please answer y, n, or s.")

    print(f"\n{len(approved)}/{len(cases)} test case(s) approved.\n")

    if not approved:
        print("[verify] No test cases were approved — falling back to the "
              "original generated tests so scoring doesn't break entirely.")
        return tests

    lines = []
    for c in approved:
        lines.append(f"def {c['name']}():")
        assert_line = f"    assert {c['func']}({c['args']}) {c['op']} {c['expected_raw']}"
        if c["msg"]:
            assert_line += f", {c['msg']}"
        lines.append(assert_line)
        lines.append("")
    return "\n".join(lines)    


# ----------------------------------------------------------------------
# Best-of-N helpers
# ----------------------------------------------------------------------
def _generate_parallel(base_prompt: str, cfg: AgentConfig) -> list:
    """Fire N differently-seeded LLM calls in parallel. ThreadPoolExecutor
    is the right tool here (not multiprocessing) because the work is
    I/O-bound: waiting on HTTP responses from Ollama/OpenAI/Anthropic."""
    prompts = [_tag_candidate(base_prompt, i) for i in range(cfg.n_candidates)]
    with ThreadPoolExecutor(max_workers=cfg.n_candidates) as pool:
        return list(pool.map(lambda p: call_llm(p, cfg), prompts))


def _evaluate_parallel(codes: list, tests: str, cfg: AgentConfig) -> list:
    """Run pytest for each candidate in parallel. Each uses its own
    tempdir so there's no cross-talk between candidates."""
    with ThreadPoolExecutor(max_workers=cfg.n_candidates) as pool:
        results = list(pool.map(lambda c: run_code(c, tests, cfg), codes))
    return list(zip(codes, results))


def _best_of(scored: list) -> tuple:
    """Pick the highest-scoring (code, result) pair; ties broken by
    shorter code as a mild proxy for simplicity."""
    return max(scored, key=lambda pair: (pair[1]["score"], -len(pair[0])))


# ----------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------
def solve(task: str, cfg: AgentConfig = AgentConfig()) -> dict:
    signature = generate_signature(task, cfg)
    print(f"\n=== AGREED SIGNATURE ===\n{signature['line']}\n")

    tests = generate_tests(task, cfg, signature)
    tests = _strip_foreign_defs(tests, signature)

    print("\n========== GENERATED TESTS ==========\n")
    print(tests)
    print("\n=====================================\n")

    with open("generated_tests.py", "w", encoding="utf-8") as f:
        f.write(tests)

    if cfg.verify_tests:
        tests = human_verify_tests(tests, task)
        print("\n========== VERIFIED TESTS (used for scoring) ==========\n")
        print(tests)
        print("\n=========================================================\n")
        with open("verified_tests.py", "w", encoding="utf-8") as f:
            f.write(tests)

    print(f"=== Round 0: generating {cfg.n_candidates} candidate solutions in parallel ===")
    candidates = _generate_parallel(_code_gen_prompt(task, signature), cfg)
    scored = _evaluate_parallel(candidates, tests, cfg)
    for i, (_, r) in enumerate(scored):
        print(f"  candidate {i}: {r['total'] - r['failed_count']}/{r['total']} passing")
        if r["total"] == 0:
            print(f"    (0 tests collected — likely a crash before pytest could run; "
                  f"stderr: {r['stderr'][:300]})")
        elif not r["passed"]:
            print(f"    failure detail: {r['stderr'][:300]}")      

    code, result = _best_of(scored)
    best = {"code": code, "score": result["score"], "result": result}
    history = [{"iteration": 0, "candidates_evaluated": cfg.n_candidates,
                "best_score": result["score"]}]

    tests_verified = cfg.verify_tests  # already verified up front if the flag was on

    for i in range(1, cfg.max_iters):
        if best["result"]["passed"]:
            print("All tests passed.")
            break

        if (not tests_verified and cfg.auto_verify_after_rounds > 0
                and i == cfg.auto_verify_after_rounds):
            print(f"\n=== Still failing after {i} round(s) (best so far: "
                  f"{best['score']:.0%}). This usually means the GENERATED "
                  f"TESTS are wrong -- e.g. contradictory expectations -- not "
                  f"the code. Pausing for human verification of the tests. ===")
            tests = human_verify_tests(tests, task)
            tests = _strip_foreign_defs(tests, signature)
            tests_verified = True
            with open("verified_tests.py", "w", encoding="utf-8") as f:
                f.write(tests)

            print("Re-scoring the best candidate so far against the corrected tests...")
            result = run_code(best["code"], tests, cfg)
            best = {"code": best["code"], "score": result["score"], "result": result}
            history.append({"iteration": i, "note": "auto-triggered human test verification",
                             "best_score": result["score"]})

            if best["result"]["passed"]:
                print("All tests passed after verification.")
                break

        print(f"\n=== Round {i}: generating {cfg.n_candidates} fix candidates in parallel ===")
        fix_prompts = _fix_prompt(task, best["code"], tests, best["result"]["stderr"], signature)
        candidates = _generate_parallel(fix_prompts, cfg)
        scored = _evaluate_parallel(candidates, tests, cfg)
        for j, (_, r) in enumerate(scored):
            print(f"  candidate {j}: {r['total'] - r['failed_count']}/{r['total']} passing")
            if r["total"] == 0:
                print(f"    (0 tests collected — likely a crash before pytest could run; "
                      f"stderr: {r['stderr'][:300]})")
            elif not r["passed"]:
                print(f"    failure detail: {r['stderr'][:300]}")      

        round_best_code, round_best_result = _best_of(scored)
        history.append({"iteration": i, "candidates_evaluated": cfg.n_candidates,
                         "best_score": round_best_result["score"]})

        if round_best_result["score"] > best["score"]:
            best = {"code": round_best_code, "score": round_best_result["score"],
                    "result": round_best_result}
    else:
        if not best["result"]["passed"]:
            print(f"Max iterations reached. Returning best attempt "
                  f"(score={best['score']:.0%}).")

    return {
        "final_code": strip_code_fence(best["code"]),
        "score": best["score"],
        "history": history,
    }


# ----------------------------------------------------------------------
# Real backend hookups (pick one, set AgentConfig(model=...))
# Each receives the candidate-tagged prompt — use the [[candidate=N]]
# prefix to vary temperature/seed per call for genuine sampling
# diversity instead of firing the same prompt N times.
# ----------------------------------------------------------------------
def call_llm_ollama(prompt: str, cfg: AgentConfig) -> str:
    import requests
    cid = _extract_candidate_id(prompt) or 0
    try:
        resp = requests.post(
            f"{cfg.ollama_url}/api/generate",
            json={
                "model": cfg.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "seed": cid},
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


def list_ollama_models(ollama_url: str = "http://localhost:11434") -> list:
    """Query Ollama for locally-installed models, e.g. to populate a
    --ollama-model choice list or sanity-check before a run."""
    import requests
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Couldn't reach Ollama at {ollama_url}. Is it running? "
            f"Start it with `ollama serve`."
        )
    return [m["name"] for m in resp.json().get("models", [])]


def call_llm_openai_compatible(prompt: str) -> str:
    import requests
    cid = _extract_candidate_id(prompt) or 0
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "your-model",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "seed": cid,
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
        "max_tokens": 1000,
        "temperature": 1.0,  # Anthropic has no seed param; temperature drives diversity
        "messages": [{"role": "user", "content": prompt}],
    })
    return resp.json()["content"][0]["text"]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
DEFAULT_TASK = (
    "Write a function `second_largest_unique(nums)` that returns the "
    "second largest unique number in a list, or None if fewer than "
    "two unique values exist."
)


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Self-correcting coding agent")
    p.add_argument("--backend", choices=["mock", "ollama", "openai", "anthropic"],
                    default="mock", help="Which LLM backend to use")
    p.add_argument("--ollama-model", default="llama3",
                    help="Local Ollama model name (e.g. llama3, qwen2.5-coder, "
                         "codellama, deepseek-coder-v2). Ignored unless --backend ollama.")
    p.add_argument("--ollama-url", default="http://localhost:11434",
                    help="Ollama server URL")
    p.add_argument("--list-ollama-models", action="store_true",
                    help="List locally-installed Ollama models and exit")
    p.add_argument("--task", default=DEFAULT_TASK, help="Coding task for the agent")
    p.add_argument("--task-file", default=None,
                    help="Path to a text file containing the task prompt "
                         "(overrides --task; supports multi-line prompts as-is)")
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--n-candidates", type=int, default=4,
                    help="Best-of-N width per round")

    p.add_argument("--verify-tests", action="store_true",
                    help="Interactively show each generated test's input -> "
                         "expected output and let you confirm/fix/skip it "
                         "before any solution is scored against it.")                
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.list_ollama_models:
        try:
            models = list_ollama_models(args.ollama_url)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)
        if not models:
            print("No models installed. Pull one with e.g. `ollama pull llama3`.")
        else:
            print("Installed Ollama models:")
            for m in models:
                print(f"  - {m}")
        sys.exit(0)

    task = args.task
    if args.task_file:
        try:
            with open(args.task_file, "r", encoding="utf-8") as f:
                task = f.read().strip()
        except FileNotFoundError:
            print(f"Error: --task-file not found: {args.task_file}")
            sys.exit(1)

    cfg = AgentConfig(
        model=args.backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        max_iters=args.max_iters,
        n_candidates=args.n_candidates,
        verify_tests=args.verify_tests,
    )

    try:
        outcome = solve(task, cfg)
    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print("\n=== FINAL CODE ===\n", outcome["final_code"])
    print(f"\n=== SCORE: {outcome['score']:.0%} ===")
    print(f"=== Rounds used: {len(outcome['history'])} ===")
