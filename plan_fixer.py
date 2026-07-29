"""
Plan Fixer (Step 2.5) — standalone module
==========================================
Takes the output of planner.py's plan() -- or a plan a human wrote by
hand -- and puts it in front of a human for review before it's treated
as ready for a Worker step.

DESIGN INTENT (deliberately human-heavy, on purpose):
  - This module does NOT auto-repair anything and does NOT call an LLM.
    Its own judgement is limited to pointing things out; a human decides
    what, if anything, actually changes.
  - detect_issues() is a dumb heuristic flashlight, not a fixer. It is
    allowed to be wrong in both directions (false positives it shouldn't
    "fix" on its own, false negatives it can't catch). That's why every
    flagged issue and the plan as a whole go through a human in
    interactive_review() before anything is finalized.
  - A human can also skip the LLM planner entirely and hand-author a
    plan (same JSON shape planner.py emits -- see load_human_plan()).
    Human-authored plans still go through the same review loop, since a
    human writing alone can still drop something a second human-in-the-
    loop pass would catch.

Flow:
  planner.plan(goal)  ─┐
                        ├─> TaskDAG ──> detect_issues(goal, dag) ──┐
  load_human_plan(path)─┘                                          │
                                                                     v
                                                     interactive_review(...)
                                                     (human approves/edits/
                                                      adds tasks, one at a
                                                      time, nothing is
                                                      auto-applied)
                                                                     │
                                                                     v
                                                        revalidated TaskDAG
"""

import os
import re
import json
from dataclasses import dataclass
from typing import List, Optional

from planner import Task, TaskDAG, PlannerConfig, plan as planner_plan


# ----------------------------------------------------------------------
# Issue
# ----------------------------------------------------------------------
@dataclass
class Issue:
    kind: str                  # short machine tag, e.g. "entity_coverage"
    message: str                # human-readable summary
    task_ids: List[str]         # tasks this issue is about (may be empty)
    detail: str = ""            # extra context to show the human


# ----------------------------------------------------------------------
# Detection -- heuristics only. Every function here is a "maybe check
# this" signal for the human, never a verdict.
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


def _entity_coverage_issues(goal: str, tasks: List[Task]) -> List[Issue]:
    """Find 'X and Y can/will <verb>' patterns in the goal (multiple
    actors tied to one action) and flag any task whose description
    plausibly matches that action but only names one of the two actors."""
    issues = []
    task_text = {t.id: f"{t.description} {t.function_name}".lower() for t in tasks}

    for m in _ACTOR_PAIR_RE.finditer(goal):
        actor_a, actor_b = m.group(1).strip().lower(), m.group(2).strip().lower()
        # look at the clause following the match for an action verb to anchor on
        tail = goal[m.end(): m.end() + 120]
        verb_match = _ACTION_VERB_RE.search(tail)
        if not verb_match:
            continue
        verb_stem = verb_match.group(1).lower()[:5]  # crude stem

        a_key = actor_a.split()[-1][:5]
        b_key = actor_b.split()[-1][:5]

        for tid, text in task_text.items():
            if verb_stem not in text:
                continue
            has_a, has_b = a_key in text, b_key in text
            if has_a != has_b:  # exactly one of the two actors present
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
    """If task B's text references a concept that another task A
    explicitly calculates/produces, but B doesn't transitively depend on
    A, flag it. Purely name-matching -- not a real data-flow check --
    so this is deliberately conservative to avoid drowning the human
    reviewer in noise:
      - producers are never flagged against each other (parallel
        "calculate X" siblings aren't consumers of one another just
        because they share a generic word like "share")
      - matching uses words unique to ONE producer's name, not words
        every producer happens to share (e.g. "share" itself)
      - a suggestion is dropped if the producer already transitively
        depends on the consumer, since that means the relationship is
        already established (or reversing it would cycle)
    """
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
        # capitalized words after the verb, e.g. calculateBlockchainShare
        # -> ["blockchain", "share"]
        return [w.lower() for w in re.findall(r"[A-Z][a-z]+", fn_name)]

    producer_modifiers = {p.id: modifier_words(p.function_name) for p in producers}

    for producer in producers:
        mods = producer_modifiers[producer.id]
        # keep only words this producer doesn't share with any other producer
        unique = [w for w in mods if len(w) >= 4 and
                  sum(1 for other in producers if w in producer_modifiers[other.id]) == 1]
        match_words = unique or mods
        if not match_words:
            continue

        for consumer in tasks:
            if consumer.id == producer.id or consumer.id in producer_ids:
                continue  # skip sibling producers entirely
            text = f"{consumer.description} {consumer.function_name}".lower()
            if not any(w in text for w in match_words):
                continue
            consumer_deps = transitive_deps(consumer.id) | {consumer.id}
            producer_deps = transitive_deps(producer.id) | {producer.id}
            if producer.id in consumer_deps:
                continue  # already satisfied
            if consumer.id in producer_deps:
                continue  # producer already depends on consumer -- wrong direction, skip
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
    # drop shorter keywords that are substrings of a longer keyword also
    # matched (e.g. "java" is contained in "javascript" -- keep only the
    # longer, more specific match)
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


_GENERIC_PIPELINE_VERBS = {"read", "extract", "deduplicate", "dedupe", "parse",
                            "fetch", "load", "scrape", "crawl", "save", "store"}


def _split_words(name: str) -> List[str]:
    """Split a function/class name into lowercase words, handling both
    snake_case and camelCase (models are inconsistent about which they use)."""
    words = []
    for part in re.split(r"[_\W]+", name):
        words += re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", part)
    return [w.lower() for w in words if w]


def _scope_drift_issues(goal: str, tasks: List[Task]) -> List[Issue]:
    """Flag tasks whose function name opens with a generic data-pipeline
    verb (read/extract/dedupe/parse/fetch/load/scrape) when that verb
    never appears anywhere in the goal text. This is a narrow, specific
    check for a known failure mode: a model defaulting to a generic
    ETL template instead of the workflow actually described. It will
    NOT catch other kinds of off-goal drift -- it only catches this one
    pattern."""
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
# Human-authored plans
# ----------------------------------------------------------------------
def load_human_plan(path: str) -> TaskDAG:
    """Load a plan a human wrote by hand. Same JSON shape as the planner
    LLM's output: {"tasks": [{"id", "description", "class_name",
    "function_name", "params", "returns", "depends_on"}, ...]}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TaskDAG.from_dict(data)


# ----------------------------------------------------------------------
# Interactive review -- nothing here is applied without an explicit
# human choice at the prompt. This is the "maximum human input" part.
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


def interactive_review(goal: str, dag: TaskDAG, issues: List[Issue]) -> TaskDAG:
    """Walk a human through every detected issue one at a time. Nothing
    is changed unless the human explicitly chooses an action. Returns a
    re-validated TaskDAG (raises if the human's edits introduce a cycle
    or dangling dependency)."""
    print(f"\n=== Plan Fixer: {len(issues)} issue(s) flagged for review ===")
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

        # Plan-wide issues (no specific task attached) can't be fixed by
        # editing one task's description -- give them their own menu
        # instead of a dead-end "which task id?" prompt.
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
                if choice in ("c", "s"):  # 's' kept as an alias for muscle memory
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
            continue
        # move to next issue after an explicit 's' (or after e/d/n if the
        # human then wants to move on -- ask each loop)
        # move to next issue after an explicit 's' -- unless the human
        # says they're not actually done, in which case redo this issue
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

    # Free-form pass: let the human add anything the heuristics missed.
    print("\n--- Free review: add any tasks the heuristics didn't catch ---")
    while _prompt("Add another task? [y/N] ").strip().lower() == "y":
        _add_task(dag)

    return _revalidate(dag)


def _revalidate(dag: TaskDAG) -> TaskDAG:
    """Rebuild the TaskDAG from its current tasks to re-run cycle/missing
    -dependency validation after human edits."""
    return TaskDAG(list(dag.tasks.values()), language=dag.language)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Plan Fixer: human-in-the-loop plan review")
    p.add_argument("--backend", choices=["mock", "ollama", "openai", "anthropic"],
                    default="mock", help="Planner LLM backend (ignored if --human-plan-file is given)")
    p.add_argument("--ollama-model", default="llama3")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--goal", default=None, help="High-level coding goal")
    p.add_argument("--goal-file", default=None, help="Path to a text file containing the goal")
    p.add_argument("--human-plan-file", default=None,
                    help="Skip the LLM planner entirely; load a hand-authored plan JSON instead")
    p.add_argument("--no-review", action="store_true",
                    help="Only print detected issues, skip the interactive review loop")
    p.add_argument("--output", default=None, help="Write the final reviewed plan as JSON to this path")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    goal = args.goal or ""
    if args.goal_file:
        with open(args.goal_file, "r", encoding="utf-8") as f:
            goal = f.read().strip()

    if args.human_plan_file:
        dag = load_human_plan(args.human_plan_file)
        if not goal:
            goal = _prompt("No --goal given for a human plan -- paste the goal text "
                            "so issue detection has something to check against: ")
    else:
        if not goal:
            goal = _prompt("Enter the goal to plan: ")
        cfg = PlannerConfig(model=args.backend, ollama_model=args.ollama_model, ollama_url=args.ollama_url)
        dag = planner_plan(goal, cfg)

    print("\n=== Plan before review ===")
    dag.pretty_print()

    issues = detect_issues(goal, dag)

    if args.no_review:
        print(f"\n=== {len(issues)} issue(s) detected (review skipped, --no-review) ===")
        for issue in issues:
            print(f"  [{issue.kind}] {issue.message}")
    else:
        dag = interactive_review(goal, dag, issues)
        print("\n=== Plan after review ===")
        dag.pretty_print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(dag.to_dict(), f, indent=2)
        print(f"\nWrote final plan to {args.output}")
