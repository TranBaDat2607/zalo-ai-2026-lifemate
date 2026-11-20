"""Final plan-3 data: the gated files with a cap on answer copies, behaviour shares checked, sizes and hours reported.

    python scripts/p3/make_mix.py [--cap 4]       # outputs/p3/final/<type>_<pool>.jsonl + outputs/p3/final/mix_report.json

T2a, T2b and T7 rows repeat a train sample's gold calls (only the heard or reworded request differs), so a sample
appears there up to 12 times: on top of the two epochs the starting adapters already did, that is mostly memorising
the same answers (log.md 12 Nov loss scan). Here every train sample keeps at most --cap such copies, chosen in order:
its avg5 reading (what the model hears at test time), its Gemini rewordings (new wording and restatement target),
its synthetic hearing (T2b), then the other real readings. T8 (same situation, one value changed: a new answer) keeps
at most --t8-cap rows per source. T3, T4, T5, T6, T9 have their own answers and are kept whole.
The report gives per type: rows, tokens (chars / 2.37, the ratio measured with the Qwen tokenizer on T2a), the share of
rows that end with an ask and with no call, against train (14.0 % / 7.2 %); T5 is trimmed when "no call" would exceed
train's share by more than 1 point.
"""
import argparse
import json
import random
from collections import Counter, defaultdict

import common

TYPES = ["t1", "t2a", "t2b", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
COPIES = {"t2a", "t2b", "t7"}
CHARS_PER_TOKEN = 2.37
ASK = "sys.ask_missing_info"
FINAL = common.OUT / "final"


def priority(r: dict) -> tuple:
    t = r["type"].lower()
    if t == "t2a":
        return (0, "") if r.get("reading") == "avg5" else (3, r.get("reading", ""))
    if t == "t7":
        return (1, r.get("style", ""))
    return (2, "")


def behaviour(r: dict) -> tuple:
    calls = r["item"]["label"]["tool_calls"]
    ask = bool(calls) and calls[-1]["name"] == ASK
    none = not calls
    return ask, none


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--t8-cap", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    FINAL.mkdir(parents=True, exist_ok=True)
    report = {}
    for pool in ["full", "fold0"]:
        rng = random.Random(f"{args.seed}:{pool}")
        rows = {}
        for t in TYPES:
            p = common.OUT / "gated" / f"{t}_{pool}.jsonl"
            if p.exists():
                rows[t] = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        # cap copies of the same answer per train sample
        by_src = defaultdict(list)
        for t in COPIES & set(rows):
            for r in rows[t]:
                by_src[r["source"]].append(r)
        keep_ids = set()
        for src, rs in by_src.items():
            rs.sort(key=lambda r: (priority(r), json.dumps(r["texts"], ensure_ascii=False)))
            keep_ids |= {id(r) for r in rs[:args.cap]}
        for t in COPIES & set(rows):
            rows[t] = [r for r in rows[t] if id(r) in keep_ids]
        if "t8" in rows:
            per, kept = Counter(), []
            for r in sorted(rows["t8"], key=lambda r: rng.random()):
                if per[r["source"]] < args.t8_cap:
                    per[r["source"]] += 1
                    kept.append(r)
            rows["t8"] = kept
        # behaviour shares of the conversations
        convs = [r for t in rows for r in rows[t] if r["kind"] == "conv"]
        asks = sum(behaviour(r)[0] for r in convs)
        nones = sum(behaviour(r)[1] for r in convs)
        if "t5" in rows and convs and nones / len(convs) > 0.072 + 0.01:
            # dropping a T5 row lowers the conversation count too: (n - e) / (N - e) <= cap  =>  e >= (n - cap N) / (1 - cap)
            cap = 0.072 + 0.01
            extra = -int(-(nones - cap * len(convs)) // (1 - cap))
            drop = set(id(r) for r in rng.sample(rows["t5"], min(extra, len(rows["t5"]))))
            rows["t5"] = [r for r in rows["t5"] if id(r) not in drop]
            convs = [r for t in rows for r in rows[t] if r["kind"] == "conv"]
            asks = sum(behaviour(r)[0] for r in convs)
            nones = sum(behaviour(r)[1] for r in convs)
        info, total_tok, total_rows = {}, 0, 0
        for t in TYPES:
            if t not in rows:
                continue
            lines = [json.dumps(r, ensure_ascii=False) for r in rows[t]]
            (FINAL / f"{t}_{pool}.jsonl").write_text("".join(x + "\n" for x in lines), encoding="utf-8")
            tok = sum(map(len, lines)) / CHARS_PER_TOKEN
            info[t] = {"rows": len(lines), "tokens_M": round(tok / 1e6, 2)}
            total_tok += tok
            total_rows += len(lines)
        for t in info:
            info[t]["token_share"] = round(info[t]["tokens_M"] * 1e6 / total_tok, 3)
        copies = Counter()
        for r in (r for t in COPIES & set(rows) for r in rows[t]):
            copies[r["source"]] += 1
        report[pool] = {"types": info, "rows": total_rows, "tokens_M": round(total_tok / 1e6, 2),
                        "conversations": len(convs), "ask_share": round(asks / max(len(convs), 1), 4),
                        "none_share": round(nones / max(len(convs), 1), 4), "train_ask_share": 0.140,
                        "train_none_share": 0.072, "answer_copies_per_sample": dict(sorted(Counter(copies.values()).items())),
                        "distinct_answers": len(convs) - sum(copies.values()) + len(copies)}
        print(json.dumps({pool: report[pool]}, ensure_ascii=False))
    (FINAL / "mix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
