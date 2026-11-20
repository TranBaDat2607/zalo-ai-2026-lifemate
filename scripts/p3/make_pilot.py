"""Pilot slices of plan3_final.md section 3, cut from the frozen fold-0 mix (outputs/p3/final/*_fold0.jsonl).

    python scripts/p3/make_pilot.py --tokens 2.0e6      # outputs/p3/pilot/{A,A2}/<type>_fold0.jsonl + pilot.json

A  : a random slice of the full fold-0 mix, every type at its token share, ~--tokens Qwen tokens.
A2 : the same size without T7 (branch A' of the plan: measures what T7 adds); the freed share goes to the other types
     in proportion.
Rows are counted in Qwen tokens exactly as train.py encodes them (encode_check's call). pilot.json gives rows, tokens
and the optimizer steps a stream run takes (extra rows / (1 - replay share) items, 4 items per step), which branch B
(replay only, --max-steps) copies.
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data, prompts  # noqa: E402

TYPES = ["t2a", "t2b", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
FINAL = REPO / "outputs" / "p3" / "final"
OUT = REPO / "outputs" / "p3" / "pilot"


def tokens_of(rows: list, cat: dict, tok, hok) -> list:
    out = []
    for k, r in enumerate(rows):
        if r["kind"] == "conv":
            ids, _ = prompts.encode_conversation(r["item"], r["texts"], cat, tok, True, 3700, 0.5, random.Random(k), hok,
                                                 None, None)
        else:
            ids, _ = prompts.encode_tool_pack(r["pack"], cat, tok)
        out.append(len(ids))
    return out


def cut(rows: dict, toks: dict, types: list, budget: float, seed: int) -> dict:
    total = {t: sum(toks[t]) for t in types}
    share = {t: total[t] / sum(total.values()) for t in types}
    out = {}
    for t in types:
        idx = list(range(len(rows[t])))
        random.Random(f"{seed}:{t}").shuffle(idx)
        got, keep = 0, []
        for i in idx:
            if got >= share[t] * budget:
                break
            keep.append(i)
            got += toks[t][i]
        out[t] = sorted(keep)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=2.0e6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--replay-share", type=float, default=0.2)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    tok = lambda t: tk(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    cat = {**data.load_catalog("public_test", data.find_root("public_test")),
           **data.load_catalog("train", data.find_root("train"))}
    hok = prompts.final_call_policy(data.load_split("train", data.find_root("train")))
    rows = {t: [json.loads(x) for x in (FINAL / f"{t}_fold0.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            for t in TYPES}
    toks = {t: tokens_of(rows[t], cat, tok, hok) for t in TYPES}
    report = {}
    for name, types in (("A", TYPES), ("A2", [t for t in TYPES if t != "t7"])):
        picked = cut(rows, toks, types, args.tokens, args.seed)
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.jsonl"):
            old.unlink()
        info = {}
        for t, idx in picked.items():
            (folder / f"{t}_fold0.jsonl").write_text("".join(json.dumps(rows[t][i], ensure_ascii=False) + "\n" for i in idx),
                                                    encoding="utf-8")
            info[t] = {"rows": len(idx), "tokens": sum(toks[t][i] for i in idx)}
        n = sum(v["rows"] for v in info.values())
        report[name] = {"types": info, "rows": n, "tokens": sum(v["tokens"] for v in info.values()),
                        "stream_steps": math.ceil(math.ceil(n / (1 - args.replay_share)) / 4)}
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "types"}))
    report["full_fold0_tokens"] = sum(sum(v) for v in toks.values())
    (OUT / "pilot.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
