"""Where each validation sample first goes off the gold path, and whether the agent could have seen it.

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python scripts/first_error.py outputs/notes-aux/aux_eval

Walks the predicted and gold calls in order (official call matching) and classifies the first divergence:
stopped early, extra calls at the end, wrong arguments (read or write; which argument kind), asked instead of acting,
acted instead of asking, a skipped gold call, an inserted extra call, or a different tool. Pred calls are replayed in
one fresh ToolEnv per sample (train split, analysis only, never at inference) to see whether a read came back empty.
"Heard ok" = the model's restated request (first think block) has at most 2 word errors against the gold transcript
(fillers such as nha/nhé and digit/word forms count as errors, so 0 is too strict: only 32 % of perfect samples have 0).
"""
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "notes"))
from compare_notes import batch, load  # noqa: E402
from lifemate import data, scorer  # noqa: E402

ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")
CLOCK = re.compile(r"^\d{1,2}:\d{2}")


def words(text: str) -> list:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"[^\w\s]", " ", text).split()


def word_errors(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def restated(row: dict) -> str:
    first = (row.get("turns") or [""])[0]
    return first.split("</think>")[0] if "</think>" in first else ""


def arg_kind(name: str, key: str, value, catalog: dict) -> str:
    spec = catalog.get(name, {}).get("parameters", {}).get("properties", {}).get(key, {})
    if spec.get("x-ref"):
        return "id"
    if spec.get("x-match") == "fuzzy":
        return "free text"
    if spec.get("enum"):
        return "choice"
    if isinstance(value, bool):
        return "choice"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str) and (ISO.match(value) or CLOCK.match(value)):
        return "date/time"
    if isinstance(value, list):
        return "list"
    return "name/text"


def wrong_args(pred: dict, gold: dict, catalog: dict) -> str:
    name = gold["name"]
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    fuzzy = scorer._fuzzy_params(name, catalog)
    kinds = set()
    for key in set(pa) | set(ga):
        if key not in pa or key not in ga:
            kinds.add("missing/extra argument")
        elif key in fuzzy:
            if scorer.token_f1(pa[key], ga[key]) <= 0.6:
                kinds.add("free text")
        elif pa[key] != ga[key]:
            kinds.add(arg_kind(name, key, ga[key], catalog))
    return " + ".join(sorted(kinds)) or "?"


def empty(result) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("ok") is False:
        return True
    lists = [v for v in result.values() if isinstance(v, list)]
    return bool(lists) and all(len(v) == 0 for v in lists)


def classify(row: dict, catalog: dict) -> tuple:
    pred, gold = row["pred"], row["gold"]
    kind = lambda c: catalog.get(c["name"], {}).get("kind", "?")
    if not gold:
        return "acted on an out-of-scope request", None
    if not pred:
        return "no call (treated as out of scope)", None
    i = j = 0
    while i < len(pred) and j < len(gold) and scorer.calls_match(pred[i], gold[j], catalog):
        i, j = i + 1, j + 1
    if i == len(pred) and j == len(gold):
        return "all calls right, order or hash differs", None
    if i == len(pred):
        missed = gold[j]
        what = "ask" if missed["name"] == "sys.ask_missing_info" else kind(missed)
        return f"stopped early (next gold call: {what})", i
    if j == len(gold):
        return f"extra calls after finishing ({kind(pred[i])})", i
    p, g = pred[i], gold[j]
    if p["name"] == g["name"]:
        return f"wrong arguments in a {kind(g)}: {wrong_args(p, g, catalog)}", i
    if p["name"] == "sys.ask_missing_info":
        return "asked instead of acting", i
    if g["name"] == "sys.ask_missing_info":
        return f"acted instead of asking ({kind(p)})", i
    if any(scorer.calls_match(p, later, catalog) for later in gold[j + 1:]):
        return f"skipped a gold call ({kind(g)})", i
    if any(scorer.calls_match(later, g, catalog) for later in pred[i + 1:]):
        return f"inserted an extra call ({kind(p)})", i
    return f"different tool (gold {kind(g)}, pred {kind(p)})", i


def main() -> None:
    run = sys.argv[1]
    rows = load(run)
    root = data.find_root("train")
    items = {s["id"]: s for s in data.load_split("train", root)}
    catalog = data.load_catalog("train", root)
    tool_env = data.import_tool_env("train", root)
    n = len(rows)
    groups = collections.defaultdict(list)
    family = collections.defaultdict(list)
    empty_reads = collections.Counter()
    for sid, row in sorted(rows.items()):
        if row["score"] > 0.999:
            continue
        label, step = classify(row, catalog)
        heard = word_errors(words(restated(row)), words(items[sid]["audio_text"])) <= 2
        groups[label].append((row, heard))
        top = label.split(" (")[0].split(":")[0]
        family[top].append((row, heard))
        if step is not None and step < len(row["pred"]) and \
                catalog.get(row["pred"][step]["name"], {}).get("kind") == "read":
            env = tool_env("train", sid, root=root)
            results = [env.call(c["name"], c.get("arguments") or {}) for c in row["pred"][: step + 1]]
            empty_reads["first off-path call is a read"] += 1
            if empty(results[-1]):
                empty_reads["... and it came back empty or failed"] += 1
                empty_reads["... then recovered (score > 0)"] += row["score"] > 0
    lost_all = sum(1 - r["score"] for r in rows.values())
    print(f"{run}: {n} samples, score {100 * (1 - lost_all / n):.2f}, imperfect {sum(len(v) for v in groups.values())}, "
          f"points lost {100 * lost_all / n:.1f}")

    def show(title: str, table: dict) -> None:
        print(f"\n{title}")
        print(f"{'first divergence':<78} {'n':>4} {'lost':>5} {'zero':>5} {'heard ok':>9} {'batch A/B/C':>12}")
        for label, items_ in sorted(table.items(), key=lambda kv: -sum(1 - r["score"] for r, _ in kv[1])):
            lost = 100 * sum(1 - r["score"] for r, _ in items_) / n
            zero = sum(r["score"] == 0 for r, _ in items_)
            heard = sum(h for _, h in items_)
            abc = collections.Counter(batch(r["id"]) for r, _ in items_)
            print(f"{label[:78]:<78} {len(items_):>4} {lost:>5.1f} {zero:>5} {heard:>9} "
                  f"{abc['A']:>4}/{abc['B']}/{abc['C']}")

    show("by family", family)
    show("detail", groups)
    print("\nreads:", dict(empty_reads))


if __name__ == "__main__":
    main()
