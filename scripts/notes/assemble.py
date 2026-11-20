"""Assemble the writer's part files into notes/batch_NNN.jsonl, keeping only notes that pass the checker.

    python scripts/notes/assemble.py 0            # batch 000: parts first, then fix files override
    python scripts/notes/assemble.py 0 --final    # same, and notes that still fail are dropped for good

The first pass (parts only) and the final result are both recorded in notes/stats.json, so the pass rate before and
after rewriting is kept. Failing notes are listed with their problems in notes/checks/batch_NNN_failures.json, which is
what the writer gets back to fix.
"""
import argparse
import json
import re

from check_notes import NOTES, check_file, check_note, world

PARTS = NOTES / "parts"


def read_jsonl(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=int)
    ap.add_argument("--final", action="store_true", help="drop the notes that still fail")
    args = ap.parse_args()
    b = args.batch
    order = json.loads((NOTES / "inputs" / "order.json").read_text(encoding="utf-8"))[b]
    validation = world()[2]
    parts = sorted(PARTS.glob(f"batch_{b:03d}_p*.jsonl"), key=lambda p: int(re.search(r"_p(\d+)", p.name).group(1)))
    fixes = sorted(PARTS.glob(f"batch_{b:03d}_fix*.jsonl"), key=lambda p: int(re.search(r"_fix(\d+)", p.name).group(1)))
    first, final = {}, {}
    for p in parts:
        for rec in read_jsonl(p):
            first.setdefault(rec["id"], rec["note"])
    final.update(first)
    for p in fixes:
        for rec in read_jsonl(p):
            final[rec["id"]] = rec["note"]
    bad_ids = [i for i in list(first) + list(final) if i in validation or i not in order]
    assert not bad_ids, f"ids outside this batch or in validation: {bad_ids}"
    first_fail = {i: check_note(i, n, b) for i, n in first.items()}
    first_fail = {i: p for i, p in first_fail.items() if p}
    fail = {i: check_note(i, n, b) for i, n in final.items()}
    fail = {i: p for i, p in fail.items() if p}
    missing = [i for i in order if i not in final]
    (NOTES / "checks").mkdir(exist_ok=True)
    (NOTES / "checks" / f"batch_{b:03d}_failures.json").write_text(
        json.dumps({i: {"note": final[i], "problems": p} for i, p in fail.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    keep = [i for i in order if i in final and i not in fail]
    complete = not missing  # a batch is saved only once every sample has a note (or with --final)
    if args.final or (complete and not fail):
        out = NOTES / f"batch_{b:03d}.jsonl"
        out.write_text("".join(json.dumps({"id": i, "note": final[i]}, ensure_ascii=False) + "\n" for i in keep),
                       encoding="utf-8")
        res = check_file(out)  # the saved file is checked again as a whole, validation ids included
        assert res["summary"]["passed"] == len(keep) and not res["summary"]["file_problems"], res["summary"]
    stats_path = NOTES / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats[f"batch_{b:03d}"] = {"samples": len(order), "written": len(final), "missing": missing,
                               "first_pass_ok": len(first) - len(first_fail), "first_pass_n": len(first),
                               "final_ok": len(keep), "dropped": sorted(fail) if args.final else [],
                               "pending_fix": sorted(fail) if not args.final else []}
    stats_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({"batch": b, "written": len(final), "first_pass_ok": len(first) - len(first_fail),
                      "now_ok": len(final) - len(fail), "failing": len(fail), "missing": len(missing),
                      "saved": bool(args.final or (complete and not fail))}))


if __name__ == "__main__":
    main()
