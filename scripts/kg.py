#!/usr/bin/env python3
"""Drive the project's Kaggle workflow: private datasets, GPU kernels, logs and outputs.

    python scripts/kg.py data                 create or version the private data dataset
    python scripts/kg.py code                 create or version the private code dataset
    python scripts/kg.py upload SLUG PATH...  create or version a private dataset holding these files or folders
    python scripts/kg.py push NAME MODULE [--from-kernel K]... [--dataset D]... [--no-gpu] [-- module args]
    python scripts/kg.py status NAME          kernel status
    python scripts/kg.py wait NAME            poll until the kernel finishes
    python scripts/kg.py collect NAME         download outputs to outputs/NAME/
    python scripts/kg.py quota                remaining weekly GPU hours
    python scripts/kg.py whoami               the account in use and the owners of its datasets
    python scripts/kg.py cli ARGS...          any other kaggle command (not auth or config) under the account

--account NAME (before the command) picks an account from ~/.kaggle/kg_accounts.json, for example
{"main": {"user": "<your-kaggle-user>"}, "second": {"user": "<another-user>", "token_file": "~/.kaggle/<token file>"}}. An account
with a token file hands that token to the Kaggle CLI in its environment; the file is read, never printed.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW_DATA = REPO / "train+public_test"
STAGE = REPO / "data" / "kaggle_stage"
KERNELS = REPO / "kernels"
KAGGLE = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
DATA_SLUG, CODE_SLUG = "lifemate-data", "lifemate-code"
# Kaggle's image can lag behind the newest model architectures.
PIP = ["transformers", "peft", "accelerate"]
ACCOUNTS = Path.home() / ".kaggle" / "kg_accounts.json"
DEFAULT_USER = os.environ.get("KAGGLE_USERNAME", "<your-kaggle-user>")  # or pick one with --account
# Kaggle CLI 2.2 prints these and still exits 0; a push can also go through with some inputs missing.
SILENT_FAILURES = ("kernel push error", "dataset creation error", "dataset version creation error", "could not be added",
                   "client error", "could not submit", "could not find")
_account = {"user": DEFAULT_USER}
_env = {k: v for k, v in os.environ.items() if k not in ("VERBOSE", "VERBOSE_OUTPUT")}  # with these the CLI prints its bearer token
# IPv4 only for the CLI: its uploads to Google storage try IPv6 first with no connect timeout, and hung for an hour when
# this machine's IPv6 route was dead (10 Nov 2026). scripts/ipv4/sitecustomize.py is loaded by those processes only.
_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent / "ipv4"), _env.get("PYTHONPATH")]))


def use_account(name: str) -> None:
    global _account
    if not name:
        return
    accounts = json.loads(ACCOUNTS.read_text(encoding="utf-8")) if ACCOUNTS.exists() else {}
    if name not in accounts:
        raise SystemExit(f"unknown account {name!r}; {ACCOUNTS} lists {sorted(accounts)}")
    _account = accounts[name]
    if _account.get("token_file"):
        for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
            _env.pop(key, None)
        _env["KAGGLE_API_TOKEN"] = Path(_account["token_file"]).expanduser().read_text(encoding="utf-8").strip()


def kaggle(*args, check=True, capture=False, strict=False) -> subprocess.CompletedProcess:
    """``strict`` shows the output and fails on the error messages the CLI prints while exiting 0."""
    res = subprocess.run([KAGGLE, "-W", *args], text=True, capture_output=capture or strict, encoding="utf-8",
                         errors="replace", env=_env)
    what = f"kaggle {args[0]} {args[1] if len(args) > 1 else ''}"
    if strict:
        text = (res.stdout or "") + (res.stderr or "")
        print(text.rstrip(), flush=True)
        found = [m for m in SILENT_FAILURES if m in text.lower()]
        if found:
            raise SystemExit(f"{what} reported {found}")
    if check and res.returncode:
        raise SystemExit(f"{what} failed\n{res.stdout or ''}{res.stderr or ''}")
    return res


def username() -> str:
    return _account["user"]


def dataset_exists(ref: str) -> bool:
    res = kaggle("datasets", "status", ref, check=False, capture=True)
    return res.returncode == 0 and "Client Error" not in res.stdout + res.stderr


def wait_ready(ref: str, timeout_s: int = 1800) -> None:
    """A new version must finish processing before a kernel can attach it."""
    end = time.time() + timeout_s
    while time.time() < end:
        out = kaggle("datasets", "status", ref, check=False, capture=True).stdout.strip()
        print(f"{ref}: {out}", flush=True)
        if re.search(r"\bready\b", out.lower()):  # the CLI prints a version warning around it, despite -W
            return
        time.sleep(10)
    raise SystemExit(f"{ref} not ready after {timeout_s} s")


def publish(folder: Path, slug: str, title: str, message: str) -> None:
    """Always private: the data package may not be shared, and nothing here passes --public."""
    ref = f"{username()}/{slug}"
    meta = {"title": title, "id": ref, "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    if dataset_exists(ref):
        kaggle("datasets", "version", "-p", str(folder), "-m", message, "-r", "zip", "-t", strict=True)
    else:
        kaggle("datasets", "create", "-p", str(folder), "-r", "zip", "-t", strict=True)
    time.sleep(30)  # the status keeps saying "ready" for the old version for a moment
    wait_ready(ref)
    print(f"https://www.kaggle.com/datasets/{ref}")


def cmd_data(args) -> None:
    folder = STAGE / DATA_SLUG
    shutil.rmtree(folder, ignore_errors=True)
    folder.parent.mkdir(parents=True, exist_ok=True)
    # Hard links: the staged copy costs no disk and the raw package stays untouched.
    subprocess.run(["cp", "-al", str(RAW_DATA), str(folder)], check=True)
    for junk in folder.rglob(".DS_Store"):
        junk.unlink()
    publish(folder, DATA_SLUG, "LifeMate data", args.message)


def cmd_code(args) -> None:
    folder = STAGE / CODE_SLUG
    shutil.rmtree(folder, ignore_errors=True)
    shutil.copytree(REPO / "src" / "lifemate", folder / "lifemate", ignore=shutil.ignore_patterns("__pycache__"))
    # Kernels log this stamp, which shows the version they really got.
    (folder / "lifemate" / "BUILD.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {args.message}\n", encoding="utf-8")
    publish(folder, CODE_SLUG, "LifeMate code", args.message)


def cmd_upload(args) -> None:
    """Stage hard links of the given files and folders, so nothing else in their folders gets uploaded."""
    folder = STAGE / args.slug
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for path in map(Path, args.paths):
        if path.is_dir():
            subprocess.run(["cp", "-al", str(path), str(folder / path.name)], check=True)
        elif path.is_file():
            os.link(path, folder / path.name)
        else:
            raise SystemExit(f"{path} does not exist")
    publish(folder, args.slug, (args.title or f"LifeMate {args.slug}")[:50], args.message)


def kernel_ref(name: str) -> str:
    return f"{username()}/lifemate-{name}"


def cmd_push(args) -> None:
    user = username()
    folder = KERNELS / args.name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    extra = [a for a in args.module_args if a != "--"]
    config = {"module": f"lifemate.{args.module}", "args": ["--out", "/kaggle/working", *extra],
              "pip": (PIP if not args.no_pip else []) + args.pip}
    code = (REPO / "scripts" / "kernel_template.py").read_text(encoding="utf-8").replace("__CONFIG__", json.dumps(config))
    code_file = f"lifemate_{args.name.replace('-', '_')}.py"
    (folder / code_file).write_text(code, encoding="utf-8")
    meta = {
        "id": kernel_ref(args.name), "title": f"lifemate {args.name}", "code_file": code_file,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": not args.no_gpu, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATA_SLUG}", f"{user}/{CODE_SLUG}", *args.dataset],
        "kernel_sources": args.from_kernel, "competition_sources": [], "model_sources": [],
    }
    if not args.no_gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"  # P100 is unsupported by the image's PyTorch
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    # The CLI cannot stop a run, so every run carries its own time limit.
    kaggle("kernels", "push", "-p", str(folder), "-t", str(int(args.hours * 3600)), strict=True)
    print(f"https://www.kaggle.com/code/{meta['id']}")


def status_of(name: str) -> str:
    return kaggle("kernels", "status", kernel_ref(name), capture=True).stdout.strip()


def cmd_status(args) -> None:
    print(status_of(args.name))


def cmd_wait(args) -> None:
    while True:
        status = status_of(args.name)
        print(time.strftime("%H:%M:%S"), status, flush=True)
        if re.search(r"complete|error|cancel", status, re.I):
            sys.exit(0 if "complete" in status.lower() else 1)
        time.sleep(args.every)


def cmd_collect(args) -> None:
    dest = REPO / "outputs" / args.name
    dest.mkdir(parents=True, exist_ok=True)
    extra = ["--file-pattern", args.pattern] if args.pattern else []
    # The listing is paged; at the default 20 per page a run with thousands of files came back with one file.
    kaggle("kernels", "output", kernel_ref(args.name), "-p", str(dest), "-o", "--page-size", "200", *extra)
    print(dest)


def cmd_quota(_) -> None:
    kaggle("quota")


def cmd_cli(args) -> None:
    """Any other Kaggle CLI command under the chosen account, e.g. ``cli datasets files user/slug``."""
    rest = [a for a in args.rest if a != "--"]
    if not rest or rest[0] in ("auth", "config"):  # these can print credentials
        raise SystemExit("cli: give a kaggle command other than auth or config")
    kaggle(*rest)


def cmd_whoami(_) -> None:
    """The configured user, and the owners Kaggle reports for this login's own datasets: they must agree."""
    res = kaggle("datasets", "list", "--mine", "--csv", capture=True)
    owners = sorted({line.split("/", 1)[0] for line in res.stdout.splitlines()[1:] if "/" in line.split(",", 1)[0]})
    print(f"account user: {username()}; owners of its datasets: {owners or 'none yet'}")
    if owners and owners != [username()]:
        raise SystemExit("the login in use belongs to another account")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--account", default=os.environ.get("KG_ACCOUNT"), help="a name from ~/.kaggle/kg_accounts.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("data", cmd_data), ("code", cmd_code)):
        p = sub.add_parser(name)
        p.add_argument("-m", "--message", default="update")
        p.set_defaults(fn=fn)
    p = sub.add_parser("upload")
    p.add_argument("slug")
    p.add_argument("paths", nargs="+")
    p.add_argument("--title", default=None)
    p.add_argument("-m", "--message", default="update")
    p.set_defaults(fn=cmd_upload)
    p = sub.add_parser("push")
    p.add_argument("name")
    p.add_argument("module", help="module under lifemate, e.g. infer")
    p.add_argument("--from-kernel", action="append", default=[], help="attach a kernel's output, e.g. OWNER/lifemate-x")
    p.add_argument("--dataset", action="append", default=[], help="attach another dataset, e.g. OWNER/some-dataset")
    p.add_argument("--pip", action="append", default=[], help="extra pip requirement")
    p.add_argument("--no-pip", action="store_true", help="skip the transformers/peft/accelerate upgrade")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--hours", type=float, default=8.0, help="time limit of the run")
    p.add_argument("module_args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_push)
    for name, fn in (("status", cmd_status), ("wait", cmd_wait), ("collect", cmd_collect)):
        p = sub.add_parser(name)
        p.add_argument("name")
        if name == "wait":
            p.add_argument("--every", type=int, default=60)
        if name == "collect":
            p.add_argument("--pattern", help="regex of files to fetch, e.g. '(\\.log$|\\.json$)'")
        p.set_defaults(fn=cmd_collect if name == "collect" else fn)
    for name, fn in (("quota", cmd_quota), ("whoami", cmd_whoami)):
        sub.add_parser(name).set_defaults(fn=fn)
    p = sub.add_parser("cli")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_cli)
    args = ap.parse_args()
    use_account(args.account)
    args.fn(args)


if __name__ == "__main__":
    os.chdir(REPO)
    main()
