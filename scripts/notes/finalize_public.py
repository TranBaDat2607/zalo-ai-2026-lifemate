"""Merge the public-test shards of a run and check the result against the final1 submission format.

    python scripts/notes/finalize_public.py outputs/notes-full/public_notes outputs/notes-full/submission.jsonl

Checks, against outputs/final1/public_avg5asr/submission.jsonl (the file that scored 66.68 on public):
- one JSON object per line, 634 lines, the same ids in the same order as public test;
- exactly the keys id, tool_calls, state_hash, in that order;
- tool_calls a list of {"name": str, "arguments": dict}; state_hash 64 hex characters;
- every detail row (seconds) present, so no shard stopped early.
It prints how many rows have the same calls as final1, for information only.
"""
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data  # noqa: E402

REFERENCE = REPO / "outputs" / "final1" / "public_avg5asr" / "submission.jsonl"


def merge_shards(run: Path) -> list:
    lines = {}
    for f in sorted(run.glob("submission.*.jsonl")):
        if f.name == "submission.jsonl":
            continue
        for x in f.read_text(encoding="utf-8").splitlines():
            if x.strip():
                lines[json.loads(x)["id"]] = x
    order = [i["id"] for i in data.load_split("public_test", data.find_root("public_test"))]
    missing = [i for i in order if i not in lines]
    if missing:
        raise SystemExit(f"{len(missing)} public ids have no prediction, first: {missing[:5]}")
    return [lines[i] for i in order]


def check(lines: list, run: Path) -> dict:
    ref = [json.loads(x) for x in REFERENCE.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [json.loads(x) for x in lines]
    assert len(rows) == len(ref) == 634, (len(rows), len(ref))
    assert [r["id"] for r in rows] == [r["id"] for r in ref], "id order differs from final1"
    for r in rows:
        assert list(r) == ["id", "tool_calls", "state_hash"], (r["id"], list(r))
        assert isinstance(r["tool_calls"], list), r["id"]
        for c in r["tool_calls"]:
            assert set(c) == {"name", "arguments"} and isinstance(c["name"], str) and isinstance(c["arguments"], dict), (r["id"], c)
        assert isinstance(r["state_hash"], str) and re.fullmatch(r"[0-9a-f]{64}", r["state_hash"]), r["id"]
    details = {}
    for f in sorted(run.glob("detail.*.jsonl")):
        for x in f.read_text(encoding="utf-8").splitlines():
            if x.strip():
                d = json.loads(x)
                details[d["id"]] = d
    assert len(details) == 634, f"{len(details)} detail rows"
    same = sum(a["tool_calls"] == b["tool_calls"] for a, b in zip(rows, ref))
    seconds = sorted(d["seconds"] for d in details.values())
    return {"rows": len(rows), "format": "same as final1", "same calls as final1": same,
            "seconds mean": round(sum(seconds) / len(seconds), 2), "seconds max": seconds[-1],
            "over 30 s (T4)": sum(s > 30 for s in seconds)}


def main() -> None:
    run, dest = Path(sys.argv[1]), Path(sys.argv[2])
    lines = merge_shards(run)
    (run / "submission.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = check(lines, run)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / "submission.jsonl", dest)
    report["written"] = str(dest)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
