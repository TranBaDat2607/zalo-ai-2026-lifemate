"""Key words wrong after restating, split by what the five out-of-fold readings say (validation, 0012), plus the same
restatement count on the probe runs' restated requests (fold models on 1,159 training samples, avg5 transcripts)."""
import collections as C, json, sys, glob
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from keyword_pool import toks, value_tokens, spans, NUMW, TIME, items, run, N, readings  # noqa: E402
from restate_effect import key_missing  # noqa: E402
from first_error import restated  # noqa: E402

five = ["exp7", "m10", "B", "s2027", "s2028"]
cls = C.Counter(); pts = C.Counter()
for sid, r in run.items():
    it = items[sid]
    G, H, R = toks(it["audio_text"]), toks(r["request"]), toks(restated(r))
    argw = set().union(*[value_tokens(c.get("arguments") or {}) for c in r["gold"]]) if r["gold"] else set()
    kr = key_missing(G, R, argw); kh = key_missing(G, H, argw)
    reads = [key_missing(G, toks(readings[k][it["audio"]]), argw) for k in five]
    first = None
    for w, n in kr.items():
        right_in = sum(1 for m in reads if m[w] < n)  # readings that have this word right
        if kh[w] < n:
            c = "heard right, broken by restating" + (" (all 5 readings right)" if right_in == 5 else "")
        elif right_in == 0:
            c = "wrong in all 5 readings"
        else:
            c = f"right in {right_in} of 5 readings"
        cls[c] += n
        first = first or c
    if first and r["score"] < 0.999:
        pts[first] += (1 - r["score"]) * 100 / N
print("\n=== key words wrong after restating (validation, 0012), by the five readings ===")
for k, v in sorted(cls.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<52} {v:>3} words   points on failed samples (first such error) {pts[k]:.2f}")

# restatement on the probe runs (fold models, training samples, avg5 transcripts)
avg5 = readings["avg5"]
st = C.Counter()
for f in glob.glob(str(REPO / "outputs/probe-fold*/probe_f*/probe.0.jsonl")):
    for line in open(f):
        row = json.loads(line)
        if row["step"] != 0:
            continue
        it = items[row["id"]]
        G, H, R = toks(it["audio_text"]), toks(avg5[it["audio"]]), toks(row["restated"])
        argw = set().union(*[value_tokens(c.get("arguments") or {}) for c in it["label"]["tool_calls"]]) if it["label"]["tool_calls"] else set()
        kh, kr = key_missing(G, H, argw), key_missing(G, R, argw)
        st["requests"] += 1; st["key wrong heard"] += sum(kh.values()); st["key wrong restated"] += sum(kr.values())
        st["fixed"] += sum((kh - kr).values()); st["broken"] += sum((kr - kh).values())
        st["requests heard key-clean but restated with a key error"] += (sum(kh.values()) == 0 and sum(kr.values()) > 0)
        st["requests heard key-clean"] += sum(kh.values()) == 0
        st["all words: heard exact"] += not spans(G, H); st["all words: restated exact"] += not spans(G, R)
print("\n=== restatement on the probe runs (6 fold models x their held-out training samples) ===")
print(dict(st))
