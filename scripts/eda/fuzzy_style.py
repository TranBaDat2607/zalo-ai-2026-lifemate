"""Style of gold free-text (fuzzy) arguments, and how our predictions miss the token-F1 > 0.6 bar."""
import collections
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import catalog, load  # noqa: E402
from errors import load_run, scorer  # noqa: E402

_, T = catalog("train")
tr = load("train")


def toks(s):
    return unicodedata.normalize("NFC", str(s).lower()).split()


def strip_punct_f1(a, b):
    f = lambda s: " ".join(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", str(s).lower())).split())
    return scorer.token_f1(f(a), f(b))


gold_vals = collections.defaultdict(list)
for s in tr:
    for c in s["label"]["tool_calls"]:
        fz = scorer._fuzzy_params(c["name"], T)
        for k in fz & set(c["arguments"]):
            gold_vals[f"{c['name']}.{k}"].append((s["id"], c["arguments"][k], s["audio_text"]))

print("== gold fuzzy fields: count, median tokens, ends with '.', starts with vocative 'X ơi', has digits, share of tokens found in the request")
for k, vs in sorted(gold_vals.items(), key=lambda x: -len(x[1]))[:25]:
    lens = [len(toks(v)) for _, v, _ in vs]
    dot = sum(str(v).rstrip().endswith((".", "!", "?")) for _, v, _ in vs)
    voc = sum(bool(re.match(r"^[^,]{1,25} ơi[,!]", str(v))) for _, v, _ in vs)
    dig = sum(bool(re.search(r"\d", str(v))) for _, v, _ in vs)
    cover = statistics.mean(len(set(toks(v)) & set(toks(t))) / max(1, len(set(toks(v)))) for _, v, t in vs)
    print(f"{k:38s} n={len(vs):3d} medtok={statistics.median(lens):4.0f} endpunct={dot/len(vs):4.2f} vocative={voc/len(vs):4.2f} digits={dig/len(vs):4.2f} in_request={cover:4.2f}")

print()
print("== examples (gold vs request) for message bodies")
for k in ("sms.send.content", "chat_ott.send_message.content", "reminders.create.content", "calendar.create_event.title", "notes.create.content"):
    for sid, v, t in gold_vals.get(k, [])[:4]:
        print(f"  {k}: {v!r}  <=  {t[:110]!r}")

if len(sys.argv) > 1:
    run = load_run(sys.argv[1])
    ratios, fails, fixed_by_punct, longer, shorter = [], 0, 0, 0, 0
    for r in run.values():
        left = list(r["gold"])
        for p in r["pred"]:
            for g in left:
                if p["name"] != g["name"]:
                    continue
                fz = scorer._fuzzy_params(g["name"], T)
                for k in fz & set(g["arguments"]) & set(p.get("arguments") or {}):
                    gv, pv = g["arguments"][k], p["arguments"][k]
                    ratios.append(len(toks(pv)) / max(1, len(toks(gv))))
                    if scorer.token_f1(pv, gv) <= 0.6:
                        fails += 1
                        fixed_by_punct += strip_punct_f1(pv, gv) > 0.6
                        longer += len(toks(pv)) > len(toks(gv))
                        shorter += len(toks(pv)) < len(toks(gv))
    print()
    print(f"pred/gold token-length ratio: median {statistics.median(ratios):.2f}, mean {statistics.mean(ratios):.2f}, n={len(ratios)}")
    print(f"fuzzy pairs below 0.6: {fails}; would pass with punctuation stripped: {fixed_by_punct}; pred longer: {longer}; pred shorter: {shorter}")
