"""Run one chain of module commands per GPU inside a single kernel, because Kaggle runs one T4x2 session at a time.

    python -m lifemate.jobs --out /kaggle/working --plan '{"0": [["transcribe", "--out", "{out}/asr"]], "1": [...]}'

Each command is a module under lifemate plus its arguments; "{out}" expands to --out. A failing command stops only the
dependent steps of its own GPU's chain, so one bad experiment does not cost the other GPU's results.

With --deadline-hours, no step starts after the deadline and a step still running then is stopped, so the kernel ends
by itself before Kaggle's time limit: a kernel Kaggle cancels cannot be attached to another one.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

START_MARGIN_S = 60  # no step starts in the last minute before the deadline


def run_chain(gpu: str, chain: list, out: str, results: dict, deadline: float = None) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    results[gpu] = []
    failed = False
    for command in chain:
        module, *args = command
        independent = module.startswith("?")  # "?module" runs even if an earlier step of the chain failed
        module = module.lstrip("?")
        if failed and not independent:
            print(f"[gpu{gpu}] skip {module}: an earlier step failed", flush=True)
            continue
        if deadline is not None and time.monotonic() > deadline - START_MARGIN_S:
            print(f"[gpu{gpu}] skip {module}: past the deadline", flush=True)
            results[gpu].append({"module": module, "exit": None, "minutes": 0.0, "deadline": True})
            failed = True
            continue
        argv = [sys.executable, "-m", f"lifemate.{module}"] + [a.replace("{out}", out) for a in args]
        started = time.monotonic()
        print(f"[gpu{gpu}] start {module} {' '.join(args)}", flush=True)
        proc = subprocess.Popen(argv, env=env, start_new_session=True)  # its own group, so its workers stop with it
        code, stopped = wait(proc, deadline, f"[gpu{gpu}] {module}")
        minutes = round((time.monotonic() - started) / 60, 1)
        print(f"[gpu{gpu}] done {module} exit={code} after {minutes} min", flush=True)
        results[gpu].append({"module": module, "exit": code, "minutes": minutes, **({"deadline": True} if stopped else {})})
        failed = bool(code)


def wait(proc, deadline: float, label: str) -> tuple:
    """(exit code, whether the deadline stopped it). SIGTERM first, then SIGKILL after a minute."""
    try:
        return proc.wait(timeout=None if deadline is None else max(1.0, deadline - time.monotonic())), False
    except subprocess.TimeoutExpired:
        print(f"{label}: deadline reached, stopping it", flush=True)
    for sig, grace in ((signal.SIGTERM, 60), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            return proc.wait(timeout=grace), True
        except subprocess.TimeoutExpired:
            continue


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", required=True, help='JSON: {"gpu index": [[module, arg, ...], ...]}')
    ap.add_argument("--deadline-hours", type=float, default=0.0, help="stop every chain this long after the start")
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan)
    results = {}
    deadline = time.monotonic() + args.deadline_hours * 3600 if args.deadline_hours else None
    threads = [threading.Thread(target=run_chain, args=(gpu, chain, args.out, results, deadline)) for gpu, chain in plan.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    (Path(args.out) / "jobs.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results), flush=True)
    if any(step["exit"] and not step.get("deadline") for chain in results.values() for step in chain):
        raise SystemExit(1)  # a step the deadline stopped is not a failure: the kernel must still end as complete


if __name__ == "__main__":
    main()
