#!/usr/bin/env python3
"""
Test harness for the micro-planner. Runs three cases against a real Ollama
instance (this actually calls the model — it needs `ollama serve` running
and the model pulled).

Usage:
    python3 test_micro_planner.py --ollama-model deepseek-coder-v2:16b

Each case checks:
  - the call succeeds and passes validation (structural correctness)
  - a couple of case-specific eyeball checks (printed for you to review,
    since correctness of *content* can't be fully automated)
"""

import argparse
import json
import sys

from micro_planner import run_micro_planner


EXISTING_SOURCE = '''
class OrderProcessor:
    def __init__(self, db, event_bus):
        self.db = db
        self.event_bus = event_bus
        self.refund_state = {}

    def get_refund_status(self, order_id):
        return self.refund_state.get(order_id, 0.0)

    def close_order(self, order_id):
        self.db.mark_closed(order_id)
        self.event_bus.emit("order.closed", order_id)
'''


def case_new_class(model, host):
    print("\n=== CASE 1: brand-new class (empty source_code) ===")
    plan = run_micro_planner(
        class_name="RateLimiter",
        big_goal="Prevent API abuse by limiting requests per user per minute",
        subtask_description="Create a new RateLimiter class with a method to check and record a request",
        source_code="",
        model=model,
        host=host,
    )
    print(json.dumps(plan, indent=2))

    method_names = {m["name"] for m in plan["methods"]}
    unexpected = method_names - {"allow_request", "check_request", "record_request", "is_allowed", "reset"}
    print(f"\n[eyeball check] method names produced: {sorted(method_names)}")
    if len(method_names) > 5:
        print("[WARN] more than 5 methods for a simple rate limiter — check for hallucinated scope creep.")
    return plan


def case_add_method(model, host):
    print("\n=== CASE 2: existing class, add one method ===")
    plan = run_micro_planner(
        class_name="OrderProcessor",
        big_goal="Add support for partial refunds on cancelled orders",
        subtask_description="Add a new method issue_partial_refund(order_id, amount) that updates refund_state, writes to db, and emits a refund.issued event",
        source_code=EXISTING_SOURCE,
        model=model,
        host=host,
    )
    print(json.dumps(plan, indent=2))

    method_names = {m["name"]: m["status"] for m in plan["methods"]}
    print(f"\n[eyeball check] methods and statuses: {method_names}")

    for existing in ("get_refund_status", "close_order"):
        if existing not in method_names:
            print(f"[FAIL] existing method '{existing}' missing from output entirely.")
        elif method_names[existing] != "existing_unchanged":
            print(f"[WARN] '{existing}' should likely be existing_unchanged, got '{method_names[existing]}'.")

    if "issue_partial_refund" not in method_names:
        print("[FAIL] new method 'issue_partial_refund' was not produced.")
    elif method_names["issue_partial_refund"] != "new":
        print(f"[WARN] 'issue_partial_refund' should be status 'new', got '{method_names['issue_partial_refund']}'.")

    return plan


def case_modify_method(model, host):
    print("\n=== CASE 3: existing class, modify one method ===")
    plan = run_micro_planner(
        class_name="OrderProcessor",
        big_goal="Improve auditability of order closures",
        subtask_description="Modify close_order so it also records the closing user_id and writes an audit log entry, in addition to its existing behavior",
        source_code=EXISTING_SOURCE,
        model=model,
        host=host,
    )
    print(json.dumps(plan, indent=2))

    close_order = next((m for m in plan["methods"] if m["name"] == "close_order"), None)
    if close_order is None:
        print("[FAIL] 'close_order' missing from output.")
    else:
        print(f"\n[eyeball check] close_order status: {close_order['status']}")
        print(f"[eyeball check] close_order side_effects: {close_order['side_effects']}")
        if close_order["status"] != "modified":
            print(f"[WARN] expected status 'modified', got '{close_order['status']}'.")
        input_names = {i["name"] for i in close_order["inputs"]}
        if "user_id" not in input_names:
            print("[WARN] expected a new 'user_id' input to appear on close_order — check inputs list.")

    other = next((m for m in plan["methods"] if m["name"] == "get_refund_status"), None)
    if other and other["status"] != "existing_unchanged":
        print(f"[WARN] unrelated method 'get_refund_status' should be untouched, got status '{other['status']}'.")

    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-model", default="deepseek-coder-v2:16b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument(
        "--case", choices=["new", "add", "modify", "all"], default="all",
        help="Run one specific case or all three.",
    )
    args = parser.parse_args()

    cases = {
        "new": case_new_class,
        "add": case_add_method,
        "modify": case_modify_method,
    }

    to_run = cases if args.case == "all" else {args.case: cases[args.case]}

    for name, fn in to_run.items():
        try:
            fn(args.ollama_model, args.ollama_host)
        except Exception as e:
            print(f"\n[ERROR] case '{name}' raised: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
