"""Shared loaders for the EDA scripts (development analysis only; the solution never reads <split>_environment)."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("LIFEMATE_DATA", REPO / "train+public_test" / "train+public_test"))


def load(split):
    with open(DATA / split / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def catalog(split):
    cat = json.loads((DATA / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))
    return cat, {t["name"]: t for a in cat["apps"] for t in a["tools"]}


def train_state(sid):
    return json.loads((DATA / "train" / "train_environment" / sid / "state.json").read_text(encoding="utf-8"))


def num(sid):
    return int(sid.split("_")[1])


def gen_by_id(sid):
    """Generator batch of a TRAIN sample by id. The boundaries are where the inputs switch (date, city, persona pool):
    A = 1-424 (November), B = 425-918 (September/October, Hà Nội, five personas), C = 919-1753. Some labelling habits
    switch a little earlier: the last calendar event with an explicit end is train_0885, and train_0896 has none."""
    n = num(sid)
    return "A" if n < 425 else ("B" if n < 919 else "C")


def public_batch(sid):
    """The same for PUBLIC samples, read off the same input switch: A = 2-173, B = 174-392, C = 393-699."""
    n = num(sid)
    return "A" if n < 174 else ("B" if n < 393 else "C")


def gen_by_input(s):
    """The same batch from visible input only: B = September/October 2026 and city Hà Nội with a 5-name persona pool
    feature-free rule; A = November with >= 10 installed apps and only core apps; else C."""
    month = s["context"]["current_time"][5:7]
    city = s["context"]["user_profile"].get("city")
    napps = len(s["context"]["installed_apps"])
    if month in ("09", "10") and city == "Hà Nội":
        return "B"
    if month == "11" and napps >= 10:
        return "A"
    return "C"
