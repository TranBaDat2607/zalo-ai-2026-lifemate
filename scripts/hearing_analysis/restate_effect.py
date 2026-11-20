"""Does the restated request (the LLM's own first step) fix or break key words, compared with what it heard?
Key word = a word of a gold argument value, a number word or a time word (keyword_pool.py). Per sample: key words
wrong in H (heard) and in R (restated), counted as multisets of gold key tokens missing from the text."""
import collections as C, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "scripts" / "notes"))
sys.path.insert(0, str(Path(__file__).parent))
from keyword_pool import toks, value_tokens, spans, NUMW, TIME, items, run, N  # noqa: E402  (runs the pool once)
from first_error import restated  # noqa: E402

def key_missing(G, X, argw):
    out = C.Counter()
    for g, x in spans(G, X):
        for w in g:
            if w in argw or w in NUMW or w in TIME:
                out[w] += 1
    return out

stats = C.Counter(); per = []
for sid, r in run.items():
    it = items[sid]
    G, H, R = toks(it["audio_text"]), toks(r["request"]), toks(restated(r))
    argw = set()
    for c in r["gold"]:
        argw |= value_tokens(c.get("arguments") or {})
    kh, kr = key_missing(G, H, argw), key_missing(G, R, argw)
    fixed = sum((kh - kr).values()); broken = sum((kr - kh).values())
    stats["key wrong in heard"] += sum(kh.values()); stats["key wrong in restated"] += sum(kr.values())
    stats["fixed by restating"] += fixed; stats["broken by restating"] += broken
    per.append(dict(id=sid, score=r["score"], fixed=fixed, broken=broken, kh=sum(kh.values()), kr=sum(kr.values())))
print("\n=== restatement effect on key words (386 validation latest requests) ===")
print(dict(stats))
b = [p for p in per if p["broken"] and not p["fixed"]]
f = [p for p in per if p["fixed"] and not p["broken"]]
print(f"samples where restating only broke key words: {len(b)}, mean score {100*sum(p['score'] for p in b)/max(1,len(b)):.1f}, points lost {sum((1-p['score'])*100/N for p in b):.2f}")
print(f"samples where restating only fixed key words: {len(f)}, mean score {100*sum(p['score'] for p in f)/max(1,len(f)):.1f}")
print(f"samples heard with all key words right: {sum(p['kh']==0 for p in per)}; of these restated with a key word wrong: {sum(p['kh']==0 and p['kr']>0 for p in per)}"
      f" (mean score {100*sum(p['score'] for p in per if p['kh']==0 and p['kr']>0)/max(1,sum(p['kh']==0 and p['kr']>0 for p in per)):.1f},"
      f" points lost {sum((1-p['score'])*100/N for p in per if p['kh']==0 and p['kr']>0):.2f})")
print(f"samples heard with a key word wrong: {sum(p['kh']>0 for p in per)}; of these restated with all key words right: {sum(p['kh']>0 and p['kr']==0 for p in per)}")
json.dump(per, open(Path(__file__).with_name("restate_rows.json"), "w"))
