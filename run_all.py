"""
run_all.py — starts the Coder Service in the background, then hands
control to the Coordinator Console in the foreground.

Ctrl-C (or 'exit'/'quit' at the task> prompt) stops the console and
shuts down the Coder Service subprocess too.

Usage (same flags as coordinator_console.py):
    python3 run_all.py --backend ollama --ollama-model deepseek-coder-v2:16b
    python3 run_all.py --backend ollama --ollama-model deepseek-coder-v2:16b --verify-tests
"""

import subprocess
import sys
import time

import requests

import coordinator_console


CODER_PORT = 8001
CODER_URL = f"http://localhost:{CODER_PORT}"

PLAN_PORT = 8002
PLAN_URL = f"http://localhost:{PLAN_PORT}"


def wait_for_service(url: str, timeout_sec: int = 20) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=1).status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def main():
    args = coordinator_console.build_arg_parser().parse_args()

    print(f"[run_all] starting Coder Service on port {CODER_PORT} ...")
    coder_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "coder_service:app",
         "--port", str(CODER_PORT), "--log-level", "warning"],
    )

    print(f"[run_all] starting Plan Controller on port {PLAN_PORT} ...")
    plan_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "plan_controller:app",
         "--port", str(PLAN_PORT), "--log-level", "warning"],
    )

    try:
        if not wait_for_service(CODER_URL):
            print(f"[run_all] Coder Service did not come up within the timeout. "
                  f"Check for errors above.")
            coder_proc.terminate()
            plan_proc.terminate()
            sys.exit(1)

        if not wait_for_service(PLAN_URL):
            print(f"[run_all] Plan Controller did not come up within the timeout. "
                  f"Check for errors above.")
            coder_proc.terminate()
            plan_proc.terminate()
            sys.exit(1)

        print(f"[run_all] Coder Service is up at {CODER_URL}")
        print(f"[run_all] Plan Controller is up at {PLAN_URL}\n")

        coordinator_console.run_console(
            backend=args.backend,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            verify_tests=args.verify_tests,
            coder_url=args.coder_url,
            plan_url=args.plan_url,
        )
    finally:
        print("[run_all] shutting down Coder Service and Plan Controller ...")
        coder_proc.terminate()
        plan_proc.terminate()
        for proc in (coder_proc, plan_proc):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
