"""On the probe runs (2,318 restated requests of training samples by fold models, avg5 transcripts):
- key words the restatement FIXED: was the gold word in some of the 5 readings, or visible in context (history,
  profile, the gold read results the agent will see), or neither (a pure guess)?
- key words it BROKE: did all 5 readings have the word right?
- key words wrong in all 5 readings (still wrong after restating): common or rare words, visible in context?"""
import collections as C, json, sys, glob
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from keyword_pool import toks, value_tokens, spans, NUMW, TIME, items, readings, freq  # noqa: E402
from restate_effect import key_missing  # noqa: E402
from lifemate.asr_eval import normal_tokens  # noqa: E402

five = ["exp7", "m10", "B", "s2027", "s2028"]
def context_words(it):
    ctx = it.get("context") or {}
    blob = json.dumps(ctx.get("user_profile") or {}, ensure_ascii=False)
    for h in it.get("history") or []:
        if h.get("role") != "user":
            blob += " " + json.dumps(h, ensure_ascii=False)
    reads = " ".join(json.dumps(r, ensure_ascii=False) for c, r in zip(it["label"]["tool_calls"], it["label"]["tool_results"]))
    return set(normal_tokens(blob)), set(normal_tokens(reads))

fx, br, still = C.Counter(), C.Counter(), C.Counter()
seen = set()
for f in sorted(glob.glob(str(REPO / "outputs/probe-fold*/probe_f*/probe.0.jsonl"))):
    for line in open(f):
        row = json.loads(line)
        if row["step"] != 0:
            continue
        it = items[row["id"]]
        G, H, R = toks(it["audio_text"]), toks(readings["avg5"][it["audio"]]), toks(row["restated"])
        calls = it["label"]["tool_calls"]
        argw = set().union(*[value_tokens(c.get("arguments") or {}) for c in calls]) if calls else set()
        kh, kr = key_missing(G, H, argw), key_missing(G, R, argw)
        rd = [toks(readings[k][it["audio"]]) for k in five]
        hist, reads = context_words(it)
        for w in (kh - kr):
            in_read = any(w in x for x in rd)
            fx["fixed: in some reading" if in_read else ("fixed: in history/profile" if w in hist else
                ("fixed: in read results" if w in reads else "fixed: pure guess"))] += 1
        for w in (kr - kh):
            n_right = sum(1 for x in rd if w in x)
            br[f"broken: right in {n_right} of 5 readings"] += 1
        if row["id"] in seen:
            continue  # count 'still wrong' once per sample (two models per fold)
        seen.add(row["id"])
        for w, n in kr.items():
            if any(w in x for x in rd):
                continue
            kind = "number" if w in NUMW else ("time" if w in TIME else "argument word/name")
            still[(kind, "common" if freq[w] >= 10 else ("rare" if freq[w] < 3 else "mid"),
                   "visible" if (w in hist or w in reads) else "not visible")] += n
print("\n=== restatement fixes (probe runs) ===");  [print(f"  {k:<32} {v}") for k, v in fx.most_common()]
print("=== restatement breaks (probe runs) ==="); [print(f"  {k:<32} {v}") for k, v in sorted(br.items())]
tot = sum(still.values())
print(f"=== key words wrong in all 5 readings and after restating, {len(seen)} training samples: {tot} ===")
for k, v in still.most_common():
    print(f"  {str(k):<52} {v}")
agg = C.Counter()
for (kind, freqc, vis), v in still.items():
    agg[kind] += v; agg[freqc] += v; agg[vis] += v
print("  totals:", dict(agg))
