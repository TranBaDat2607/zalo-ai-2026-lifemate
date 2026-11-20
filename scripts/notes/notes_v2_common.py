"""Worked notes v2: which kind (K0-K4) a train sample is, and what the branch-blind writer may see of its gold calls.

    python scripts/notes/notes_v2_common.py            # counts by batch and kind (A = ids < 425, B < 919, C otherwise)
    python scripts/notes/notes_v2_common.py --show K2  # ten random samples of a kind

Kinds (see notes_v2/PROMPT.md, rule R1), decided from the gold calls of the final turn, the catalog kind of each tool
(read / write / meta), the `x-ref` parameters, and the request text:

- K0 explicit: no read of this turn decides anything. A read-only answer, a fixed plan, an ask knowable before any read,
  or nothing.
- K1 pick one item: a read must identify one item. A later call takes an ID (an `x-ref` parameter, or any ID-like value)
  from the read, or the answer asks for such an ID after a read.
- K2 condition on a read: the request has a condition ("nếu", "thì", "xem ... có ...") and a read of this turn is
  there to decide it.
- K3 select by a rule: a read lists items and the same write tool is applied to two or more of them, or the request says
  "tất cả / hết / mọi".
- K4 value lookup: a read only supplies a value for an action that happens anyway (an amount, steps, a price, a time).

A sample can have several kinds (`kinds`); `primary` is the strongest in the order K2, K3, K1, K4, K0. `cut` is the number
of gold calls the writer may see: the calls up to and including the FIRST read (K1-K3: a later read may belong to one
branch only), or all of them (K0, K4).
"""
import argparse
import collections
import json
import random
import re

from notes_common import classify, contains_token, is_id_like, leaf_values, load_train

BATCH_A, BATCH_B = 425, 919  # first id of B and of C

# a condition the request puts on something that a read of this turn can tell
_PRED = (r"mưa|nắng|hơn|trên|dưới|quá|vượt|thấp|cao|ít|nhiều|đủ|thiếu|hết|còn|chưa|rảnh|trống|trễ|muộn|lố|âm|hạn|"
         r"trúng|về|nói|bảo|báo|ra|tới|đến|gửi|thấy|đang|sắp|có|không|lạnh|nóng|giảm|tăng|bằng|yếu|đầy|rồi|bận|kẹt|tắc|xong|mở|tắt|khoá|khóa|mới|cũ|đã")
_COND = re.compile(
    r"(?<!\w)(nếu|nhược bằng|hễ|miễn là|trường hợp|còn không|không thì|nhỡ|"
    r"(?:" + _PRED + r")\b[^.?!]{0,60}?\bthì)(?!\w)", re.I)
_ALL = re.compile(r"(?<!\w)(tất cả|toàn bộ|hết|mọi|cả \d|cả hai|cả ba|từng|các)(?!\w)", re.I)
HIDDEN = "⟨từ kết quả gọi trước trong lượt này⟩"


def batch_of(sample_id: str) -> str:
    n = int(sample_id.split("_")[1])
    return "A" if n < BATCH_A else "B" if n < BATCH_B else "C"


def tool_kind(catalog: dict, name: str) -> str:
    return (catalog.get(name) or {}).get("kind", "meta")


def analyze(item: dict, catalog: dict) -> dict:
    calls = item["label"]["tool_calls"]
    results = item["label"]["tool_results"]
    request = item["audio_text"]
    facts = classify(item, catalog)
    vis = facts["visible"]
    kinds_of_call = [tool_kind(catalog, c["name"]) for c in calls]
    n = len(calls)
    info = {"id": item["id"], "batch": batch_of(item["id"]), "n_calls": n,
            "shape": "".join({"read": "R", "write": "W"}.get(k, "A") for k in kinds_of_call)}
    reads = [k for k in range(n) if kinds_of_call[k] == "read"]
    # --- ID dependencies: call k takes an ID from the result of an earlier call (and the history does not show it)
    id_dep = {}  # k -> (param, value, index of the read it comes from)
    for k, call in enumerate(calls):
        if call["name"] == "sys.ask_missing_info":
            continue
        props = (catalog.get(call["name"], {}).get("parameters") or {}).get("properties") or {}
        for key, value, spec in leaf_values(call.get("arguments") or {}, props):
            if not (isinstance(value, str) and is_id_like(value, spec)) or contains_token(vis, value):
                continue
            src = [j for j in range(k) if value in json.dumps(results[j], ensure_ascii=False)]
            if src:
                id_dep[k] = (key, value, src[-1])
                break
    asks = [k for k in range(n) if calls[k]["name"] == "sys.ask_missing_info"]
    ask_after_read = [k for k in asks if any(j < k and kinds_of_call[j] == "read" for j in range(n))]
    ask_ref = []  # asks that ask for an item ID after a read
    for k in ask_after_read:
        tool = (calls[k].get("arguments") or {}).get("tool_name")
        props = (catalog.get(tool, {}).get("parameters") or {}).get("properties") or {}
        asked = (calls[k].get("arguments") or {}).get("params") or []
        if any(props.get(p, {}).get("x-ref") or p.endswith("_id") for p in asked):
            ask_ref.append(k)
    write_idx = [k for k in range(n) if kinds_of_call[k] == "write"]
    cond = bool(_COND.search(request))
    kinds, deciding = set(), set()
    decision = None  # (write tool, param) the read must supply, for K1 notes
    # K3: the same write tool on two or more items of one list
    by_src = collections.defaultdict(list)
    for k, (key, value, j) in id_dep.items():
        if kinds_of_call[k] == "write":
            by_src[(calls[k]["name"], j)].append(k)
    k3 = any(len(v) >= 2 and len({id_dep[x][1] for x in v}) >= 2 for v in by_src.values())
    if k3:
        kinds.add("K3")
        for (_, j), v in by_src.items():
            if len(v) >= 2:
                deciding.add(j)
    # K2: a condition in the request and a read of this turn
    cond_reads = [j for j in reads if any(k > j for k in range(n)) or True]
    if cond and reads:
        # the read must come before something it could gate, or be the last call of an answer that stops there
        kinds.add("K2")
        deciding |= set(reads)
    # K1: an ID taken from a read, or an ask for an ID after a read
    for k, (key, value, j) in id_dep.items():
        if f"K3" in kinds and any(k in v for v in by_src.values() if len(v) >= 2):
            continue
        kinds.add("K1")
        deciding.add(j)
        if decision is None:
            decision = (calls[k]["name"], key)
    for k in ask_ref:
        kinds.add("K1")
        deciding |= {j for j in reads if j < k}
        a = calls[k].get("arguments") or {}
        if decision is None or True:
            decision = (a.get("tool_name"), (a.get("params") or [""])[0])
    # K4: a value from a read feeds a write that happens anyway
    if not kinds:
        opt = [e for e in facts["optional"] if e[0] != "ask"]
        if (opt or _value_from_read(calls, results, catalog, vis)) and reads and write_idx:
            kinds.add("K4")
    # a read that precedes a write which does not depend on it, with nothing else going on, stays K0
    if not kinds:
        kinds.add("K0")
    order = ["K2", "K3", "K1", "K4", "K0"]
    primary = next(k for k in order if k in kinds)
    if kinds & {"K1", "K2", "K3"}:
        # Only the FIRST read is shown. A later read can itself belong to one branch only (a second lookup happens only
        # after the first found its item, or only when the condition held), so showing it would tell the branch.
        cut = (min(reads) + 1) if reads else n
    else:
        cut = n
    info.update({"kinds": sorted(kinds), "primary": primary, "cut": cut, "deciding": sorted(deciding),
                 "decision": decision, "cond": cond, "all_word": bool(_ALL.search(request)),
                 "id_dep": {k: v[:2] for k, v in id_dep.items()}, "asks": asks,
                 "ends_ask": bool(n and calls[-1]["name"] == "sys.ask_missing_info"),
                 "any_write": bool(write_idx), "n_reads": len(reads),
                 "branch_ask": bool(n and calls[-1]["name"] == "sys.ask_missing_info"),
                 "branch_write": bool(write_idx)})
    return info


def _value_from_read(calls, results, catalog, vis) -> bool:
    """A write argument (text or number) that an earlier result of this turn holds and the model has not seen."""
    for k, call in enumerate(calls):
        if tool_kind(catalog, call["name"]) != "write" or k == 0:
            continue
        before = json.dumps(results[:k], ensure_ascii=False)
        props = (catalog.get(call["name"], {}).get("parameters") or {}).get("properties") or {}
        for key, value, spec in leaf_values(call.get("arguments") or {}, props):
            if isinstance(value, bool) or value is None or spec.get("enum"):
                continue
            text = str(value)
            if isinstance(value, str) and len(text) < 4:
                continue
            if isinstance(value, (int, float)) and abs(value) < 10:
                continue
            if text in before and not contains_token(vis, text):
                return True
    return False


def blind_kinds(info: dict) -> bool:
    return bool(set(info["kinds"]) & {"K1", "K2", "K3"})


def analyze_all():
    items, catalog, validation = load_train()
    pool = [i for i in items if i["id"] not in validation]
    assert len(validation) == 386 and len(pool) == 1159
    return pool, catalog, {i["id"]: analyze(i, catalog) for i in pool}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    pool, catalog, infos = analyze_all()
    cnt = collections.Counter((i["batch"], i["primary"]) for i in infos.values())
    print("primary kind by batch")
    for kind in ["K0", "K1", "K2", "K3", "K4"]:
        row = {b: cnt[(b, kind)] for b in "ABC"}
        print(kind, row, sum(row.values()))
    mixed = collections.Counter(tuple(i["kinds"]) for i in infos.values())
    print("kind sets:", dict(mixed))
    print("blind (K1-K3 present):", sum(blind_kinds(i) for i in infos.values()))
    if args.show:
        rnd = random.Random(args.seed)
        cands = [it for it in pool if infos[it["id"]]["primary"] == args.show]
        for it in rnd.sample(cands, min(args.n, len(cands))):
            i = infos[it["id"]]
            print("\n==", it["id"], i["batch"], i["kinds"], "shape", i["shape"], "cut", i["cut"], "decision", i["decision"])
            for t in it["history"][-2:]:
                if t["role"] == "user":
                    print("  H:", t["audio_text"][:120])
            print("  REQ:", it["audio_text"])
            for c in it["label"]["tool_calls"]:
                print("   ", c["name"], json.dumps(c["arguments"], ensure_ascii=False)[:150])


if __name__ == "__main__":
    main()
