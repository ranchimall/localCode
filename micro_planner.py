#!/usr/bin/env python3
"""
Micro-Planner: breaks ONE class into its atomic elements (methods + members),
each fully specified with inputs, outputs, returns, and side effects.

Standalone module. Not wired into the coordinator/planner/coder loop yet —
run it directly to generate and validate a plan for a single class.

Usage:
    python3 micro_planner.py \
        --class-name OrderProcessor \
        --big-goal "Add support for partial refunds on cancelled orders" \
        --subtask "Update OrderProcessor to track refund state and expose a method to issue a partial refund" \
        --source-file order_processor.py \
        --ollama-model deepseek-coder-v2:16b

If --source-file is omitted, the class is treated as new (being designed from
scratch rather than modified).
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-coder-v2:16b"
MAX_RETRIES = 3

REQUIRED_METHOD_FIELDS = [
    "id", "name", "kind", "status", "inputs", "outputs",
    "returns", "side_effects", "depends_on",
]
VALID_STATUSES = {"new", "modified", "existing_unchanged", "removed"}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(class_name, big_goal, subtask_description, source_code, violation_note=None):
    source_block = source_code if source_code.strip() else "(class does not exist yet — design it from scratch)"

    violation_block = ""
    if violation_note:
        violation_block = (
            "YOUR PREVIOUS OUTPUT WAS INVALID. FIX THIS SPECIFIC PROBLEM AND RETRY:\n"
            f"{violation_note}\n\n"
        )

    prompt = f"""You are a micro-planner. You take exactly ONE class and break it into its
atomic elements: every method and every member, each fully specified.

{violation_block}INPUT:
class_name: {class_name}
big_goal: {big_goal}
subtask_description: {subtask_description}
current_source_code:
{source_block}

RULES YOU MUST FOLLOW EXACTLY:
1. Output ONLY valid JSON matching the schema below. No prose, no markdown
   fences, no explanation before or after. The response must start with {{
   and end with }}.
2. List EVERY method and member currently in the class, even ones this
   subtask does not change. Mark unchanged ones as "existing_unchanged".
3. For anything new or modified, mark status "new" or "modified".
4. Every method must have: id, name, kind, status, inputs, outputs,
   returns, side_effects, depends_on. NONE of these fields may be omitted.
   If a field doesn't apply, use an empty array [] or the string "none" —
   never leave it out.
5. side_effects must be exhaustive: state mutation, I/O, DB writes,
   network calls, event emission, logging, global/singleton access —
   all of it, named explicitly. If there are truly none, write ["none"].
6. Do not invent behavior not implied by big_goal, subtask_description,
   or the existing source. Do not add speculative methods "for later".
7. Assign IDs sequentially per kind (mem_0001, mem_0002, meth_0001, ...).
   Preserve IDs for unchanged methods across runs where possible by
   matching on name.
8. This is ONE class only. Never reference or plan another class.

OUTPUT SCHEMA (structure to match exactly):
{{
  "class_name": "string",
  "class_id": "string (snake_case of class name, prefixed cls_)",
  "generated_from": {{
    "big_goal": "string",
    "subtask_description": "string"
  }},
  "members": [
    {{
      "id": "mem_00NN",
      "name": "string",
      "kind": "member",
      "type": "string",
      "description": "string",
      "side_effects": "string or 'none'"
    }}
  ],
  "methods": [
    {{
      "id": "meth_00NN",
      "name": "string",
      "kind": "method",
      "status": "new | modified | existing_unchanged | removed",
      "inputs": [{{"name": "string", "type": "string", "required": true}}],
      "outputs": [],
      "returns": {{"type": "string", "description": "string"}},
      "side_effects": ["string", "..."],
      "depends_on": ["id", "..."],
      "notes": "string"
    }}
  ]
}}
"""
    return prompt


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def call_ollama(prompt, model, host=DEFAULT_OLLAMA_HOST, timeout=300):
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",          # ask Ollama to constrain to JSON output
        "options": {"temperature": 0.1},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {url}. Is `ollama serve` running? ({e})"
        ) from e
    return body.get("response", "")


def extract_json(raw_text):
    """Ollama with format=json should already return clean JSON, but this
    strips markdown fences or stray text defensively, in case the model
    wraps its output anyway."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return text[start:end + 1]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(plan, source_code):
    """Returns (ok: bool, violation: str or None)."""
    if not isinstance(plan, dict):
        return False, "Top-level output must be a JSON object."

    for key in ("class_name", "members", "methods"):
        if key not in plan:
            return False, f"Top-level key '{key}' is required and missing."

    seen_ids = set()

    for m in plan.get("members", []):
        if "id" not in m or "name" not in m:
            return False, "Every member must have at least 'id' and 'name'."
        if m["id"] in seen_ids:
            return False, f"Duplicate id found: {m['id']}"
        seen_ids.add(m["id"])

    method_names = set()
    for meth in plan.get("methods", []):
        missing = [f for f in REQUIRED_METHOD_FIELDS if f not in meth]
        if missing:
            return False, (
                f"Method '{meth.get('name', '?')}' is missing required "
                f"field(s): {missing}"
            )
        if meth["id"] in seen_ids:
            return False, f"Duplicate id found: {meth['id']}"
        seen_ids.add(meth["id"])

        if meth["status"] not in VALID_STATUSES:
            return False, (
                f"Method '{meth['name']}' has invalid status "
                f"'{meth['status']}'. Must be one of {sorted(VALID_STATUSES)}."
            )

        se = meth["side_effects"]
        if se in ([], "", None):
            return False, (
                f"Method '{meth['name']}' has an empty side_effects field — "
                f"must be [\"none\"] if truly none, never empty."
            )

        method_names.add(meth["name"])

    # cross-reference depends_on against known ids
    for meth in plan.get("methods", []):
        for dep in meth.get("depends_on", []):
            if dep not in seen_ids:
                return False, (
                    f"Method '{meth['name']}' has depends_on referencing "
                    f"unknown id '{dep}'."
                )

    # naive check: every method already in source must appear in output
    if source_code and source_code.strip():
        existing_defs = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", source_code)
        for name in existing_defs:
            if name == "__init__":
                continue
            if name not in method_names:
                return False, (
                    f"Method '{name}' exists in current_source_code but is "
                    f"missing from the output. Every existing method must be "
                    f"listed, even as 'existing_unchanged'."
                )

    return True, None


# ---------------------------------------------------------------------------
# Main entry point (callable + CLI)
# ---------------------------------------------------------------------------

def run_micro_planner(
    class_name,
    big_goal,
    subtask_description,
    source_code="",
    model=DEFAULT_MODEL,
    host=DEFAULT_OLLAMA_HOST,
    max_retries=MAX_RETRIES,
    verbose=True,
):
    violation_note = None
    last_raw = None

    for attempt in range(1, max_retries + 1):
        prompt = build_prompt(
            class_name, big_goal, subtask_description, source_code, violation_note
        )
        if verbose:
            print(
                f"[micro-planner] attempt {attempt}/{max_retries} — "
                f"calling {model} via Ollama...",
                file=sys.stderr,
            )

        raw = call_ollama(prompt, model, host)
        last_raw = raw

        try:
            json_text = extract_json(raw)
            plan = json.loads(json_text)
        except (ValueError, json.JSONDecodeError) as e:
            violation_note = f"Output was not valid JSON: {e}"
            if verbose:
                print(f"[micro-planner] invalid JSON, retrying: {e}", file=sys.stderr)
            continue

        ok, violation = validate(plan, source_code)
        if ok:
            if verbose:
                print(
                    f"[micro-planner] valid plan produced on attempt {attempt}.",
                    file=sys.stderr,
                )
            return plan

        violation_note = violation
        if verbose:
            print(f"[micro-planner] validation failed, retrying: {violation}", file=sys.stderr)

    raise RuntimeError(
        f"micro-planner failed after {max_retries} attempts.\n"
        f"Last violation: {violation_note}\n"
        f"Last raw output:\n{last_raw}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Micro-planner: break one class into atomic elements."
    )
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--big-goal", required=True)
    parser.add_argument("--subtask", required=True, dest="subtask_description")
    parser.add_argument(
        "--source-file", default=None,
        help="Path to existing source file containing the class, if any.",
    )
    parser.add_argument("--backend", default="ollama", choices=["ollama"])
    parser.add_argument("--ollama-model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument(
        "--out", default=None,
        help="Path to write output JSON. Defaults to stdout.",
    )
    args = parser.parse_args()

    source_code = ""
    if args.source_file:
        with open(args.source_file, "r") as f:
            source_code = f.read()

    plan = run_micro_planner(
        class_name=args.class_name,
        big_goal=args.big_goal,
        subtask_description=args.subtask_description,
        source_code=source_code,
        model=args.ollama_model,
        host=args.ollama_host,
        max_retries=args.max_retries,
    )

    output = json.dumps(plan, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Wrote plan to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
