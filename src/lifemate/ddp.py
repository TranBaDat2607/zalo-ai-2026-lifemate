"""Run one lifemate module as several processes, one per GPU, that train one model together (lifemate.stream).

    python -m lifemate.ddp train --out {out}/m --extra ...        # in a jobs.py plan: {"0,1": [["ddp", "train", ...]]}
    python -m lifemate.ddp --nproc 2 train ...                     # CPU (tests): two processes, no GPU

Each process sees one device (CUDA_VISIBLE_DEVICES=<its GPU>) and gets RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT.
If one process fails the others are stopped. When every process exits with stream.RESTART_EXIT (they saved their state
because RAM grew), all of them are started again and resume from that state.
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time

from .stream import RESTART_EXIT


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(module: str, args: list, devices: list) -> int:
    port = free_port()
    procs = []
    for rank, dev in enumerate(devices):
        env = {**os.environ, "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(len(devices)),
               "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "CUDA_VISIBLE_DEVICES": dev}
        name = module if "." in module else f"lifemate.{module}"  # a dotted name runs as it is (tests)
        procs.append(subprocess.Popen([sys.executable, "-m", name, *args], env=env))

    def stop(*_):
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    while True:
        codes = [p.poll() for p in procs]
        failed = [c for c in codes if c not in (None, 0, RESTART_EXIT)]
        if failed:
            print(f"[ddp] a process failed (exits {codes}); stopping the others", flush=True)
            stop()
            for p in procs:
                try:
                    p.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    p.kill()
            return failed[0]
        if all(c is not None for c in codes):
            if len(set(codes)) > 1:  # one restarting and one done cannot happen with synced flags; treat it as a failure
                print(f"[ddp] processes disagree (exits {codes})", flush=True)
                return 1
            return codes[0]
        time.sleep(2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=0, help="processes; default: one per device in CUDA_VISIBLE_DEVICES")
    ap.add_argument("--max-restarts", type=int, default=8)
    ap.add_argument("module")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    devices = visible[:a.nproc] if a.nproc and visible else visible or [""] * max(a.nproc, 1)
    if a.nproc and len(devices) < a.nproc:
        devices = [""] * a.nproc  # CPU processes
    print(f"[ddp] {a.module} on {len(devices)} processes, devices {devices}", flush=True)
    for attempt in range(a.max_restarts + 1):
        code = launch(a.module, a.args, devices)
        if code != RESTART_EXIT:
            return code
        print(f"[ddp] all processes asked for a restart ({attempt + 1})", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
