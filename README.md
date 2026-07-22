# localCode
Agent for local LLM code creation and testing

# Self-Correcting Coding Agent

A minimal, single-file agent loop that writes code, writes tests for it, runs
both in a sandboxed subprocess, and iterates on failures — using **best-of-N
parallel sampling** at every round instead of generating one attempt at a
time.

Works out of the box with a built-in mock backend (no LLM required to try
it), and plugs into a local **Ollama** model with one flag.

```
=== Round 0: generating 4 candidate solutions in parallel ===
  candidate 0: 4/5 passing
  candidate 1: 5/5 passing
  candidate 2: 3/5 passing
  candidate 3: 5/5 passing
All tests passed.

=== FINAL CODE ===
 def second_largest_unique(nums):
    uniq = sorted(set(nums), reverse=True)
    if len(uniq) < 2:
        return None
    return uniq[1]

=== SCORE: 100% ===
=== Rounds used: 1 ===
```

## How it works

1. **Tests first.** The agent asks the LLM to write `test_case_1`,
   `test_case_2`, ... functions for the task before any code exists.
2. **Best-of-N generation.** Instead of one code attempt, it fires `N`
   differently-seeded LLM calls in parallel (`ThreadPoolExecutor`, since the
   bottleneck is I/O — waiting on HTTP responses — not CPU).
3. **Parallel scoring.** Each candidate is run against the tests via `pytest`
   with JUnit XML output, in its own temp directory, giving a **partial-credit
   score** (e.g. `4/5` passing) rather than a single pass/fail bit.
4. **Keep the best, not the last.** The highest-scoring candidate survives to
   the next round; a regression never overwrites a better earlier attempt.
5. **Repeat as a fix loop.** If nothing passes fully, the agent generates `N`
   fix attempts from the current best candidate + its test failures, scores
   those, and repeats up to `--max-iters` times.

## Requirements

```bash
pip install requests pytest
```

Python 3.9+. Works on Linux, macOS, and Windows.

## Quick start (no LLM needed)

```bash
python coding_agent.py --backend mock
```

Runs a self-contained demo task with a fake backend that simulates realistic
sampling variance (some candidates buggy, some correct), so you can see the
best-of-N and fix-loop mechanics without any setup.

## Using a local model via Ollama

```bash
# 1. Make sure Ollama is running and you have a model pulled
ollama pull deepseek-coder-v2:16b     # or llama3.1, qwen2.5-coder, codellama, ...

# 2. Confirm the script can see it
python coding_agent.py --list-ollama-models

# 3. Run the agent against it
python coding_agent.py --backend ollama --ollama-model deepseek-coder-v2:16b

# 4. Swap models or point at a remote Ollama instance any time
python coding_agent.py --backend ollama --ollama-model qwen2.5-coder --ollama-url http://192.168.1.50:11434
```

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--backend` | `mock` | `mock` \| `ollama` \| `openai` \| `anthropic` |
| `--task` | *(second-largest-unique demo)* | The coding task to hand the agent |
| `--max-iters` | `3` | Max rounds (round 0 + fix rounds) before giving up |
| `--n-candidates` | `4` | Best-of-N width — how many candidates generated per round |
| `--ollama-model` | `llama3` | Any locally-pulled Ollama model name, including `:tag` |
| `--ollama-url` | `http://localhost:11434` | Ollama server address |
| `--list-ollama-models` | — | List locally-installed Ollama models and exit |

## Other backends

`openai` targets any OpenAI-compatible server (e.g. **vLLM**) at
`localhost:8000`; `anthropic` calls the Claude API and expects
`ANTHROPIC_API_KEY` in your environment. Both are stubbed with working
request code in `call_llm_openai_compatible()` / `call_llm_anthropic()` —
edit the URL/model/env var to match your setup.

## Sandboxing — what it does and doesn't protect against

Generated code runs as a real Python subprocess with:

- A timeout (`--timeout`, default 5s) to kill infinite loops.
- On **Linux/macOS only**: `RLIMIT_CPU` and `RLIMIT_AS` to cap CPU time and
  memory (the `resource` module doesn't exist on Windows, so these are a
  no-op there — your only real safety net on Windows is the timeout).
- A scrubbed environment that keeps only what's needed to launch Python
  (plus OS-required vars like `SystemRoot` on Windows) — proxy settings,
  API keys, etc. in your normal shell are not inherited by generated code.

**What this is not:** a real security boundary. It does not stop filesystem
writes or network access. If you're ever running code from a source you
don't trust — not your own local model's output — put this inside a
container (Docker/gVisor/Firecracker) with no network and a disposable
filesystem instead of relying on process-level limits alone.

## Troubleshooting

**`ModuleNotFoundError: No module named 'resource'`**
Fixed as of this version — `resource` is now imported lazily and only on
POSIX. If you still see this, you're on an old copy of the script.

**Every candidate shows `0/0 passing`**
Means `pytest` crashed before writing a results file — the printed line
under each candidate now includes the actual subprocess `stderr` to show
why. The most common historical cause (fixed here) was an overly-stripped
subprocess environment breaking Python's ability to even start on Windows.

**`Couldn't reach Ollama at http://localhost:11434`**
Ollama isn't running. On Windows it usually runs as a background/tray app;
on Linux/macOS start it with `ollama serve`.

**`Model '<name>' isn't pulled`**
Run `ollama pull <name>`, or check exact installed names (including the
`:tag` suffix, e.g. `deepseek-coder-v2:16b`) with `--list-ollama-models`.

## Extending

- **Best-of-N width vs. cost:** `--n-candidates` trades LLM calls for fewer
  rounds. Cheap/local models can go high; metered APIs should stay low.
- **Real sampling diversity:** each call is tagged with `[[candidate=N]]`
  in the prompt, mapped to a `seed`/`temperature` in the backend functions —
  wire this up further if your backend supports better diversity controls.
- **True isolation:** swap `run_code()`'s subprocess call for a Docker
  `exec` call if you need to run genuinely untrusted code.



