"""What a 'readings view' would look like: the avg5 text plus the alternatives the five single readings offer where
they disagree with it. Over all 5,036 train clips (out-of-fold readings), latest requests and earlier turns:
how many clips show alternatives, how many spans, how long the view gets, and how often a key word that avg5 got wrong
is offered among the alternatives."""
import collections as C, difflib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from keyword_pool import toks, value_tokens, NUMW, TIME, items, readings, val  # noqa: E402
from restate_effect import key_missing  # noqa: E402
from lifemate.asr_eval import normal_tokens  # noqa: E402

five = ["exp7", "m10", "B", "s2027", "s2028"]

def view(main_words, others):
    """{(i1, i2): set of alternative strings} over main_words positions, overlapping ranges merged."""
    ranges = []
    for o in others:
        sm = difflib.SequenceMatcher(a=main_words, b=o, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                ranges.append([i1, i2, " ".join(o[j1:j2])])
    ranges.sort()
    merged = []
    for i1, i2, alt in ranges:
        if merged and i1 <= merged[-1][1] and not (i1 == i2 == merged[-1][1] and merged[-1][0] < merged[-1][1]):
            merged[-1][1] = max(merged[-1][1], i2); merged[-1][2].add(alt)
        else:
            merged.append([i1, i2, {alt}])
    return merged

st = C.Counter(); lens = []
for sid, it in items.items():
    clips = [(it["audio"], it["audio_text"], True)]
    for h in it.get("history") or []:
        if h.get("role") == "user" and h.get("audio") and h.get("audio_text"):
            clips.append((h["audio"], h["audio_text"], False))
    calls = it["label"]["tool_calls"]
    argw = set().union(*[value_tokens(c.get("arguments") or {}) for c in calls]) if calls else set()
    for audio, gold, latest in clips:
        if audio not in readings["avg5"]:
            continue
        main = toks(readings["avg5"][audio])
        others = [toks(readings[k][audio]) for k in five if audio in readings[k]]
        v = view(main, others)
        tag = "latest" if latest else "earlier"
        st[f"{tag} clips"] += 1
        st[f"{tag} clips with alternatives"] += bool(v)
        st[f"{tag} alternative spans"] += len(v)
        extra = sum(len(" | ".join(sorted(a)).split()) for _, _, a in v)
        lens.append((tag, len(main), extra))
        if latest:
            G = toks(gold)
            km = key_missing(G, main, argw)
            alt_words = set(w for _, _, a in v for s in a for w in s.split())
            for w, n in km.items():
                st["latest key words wrong in avg5"] += n
                st["  ...offered among the alternatives"] += n if w in alt_words else 0
            if sid in val:
                st["validation latest clips"] += 1
                st["validation latest with alternatives"] += bool(v)
for k, v in st.items():
    print(f"{k:<44} {v}")
for tag in ("latest", "earlier"):
    L = [x for x in lens if x[0] == tag]
    print(f"{tag}: mean words {sum(x[1] for x in L)/len(L):.1f}, mean extra words in the view {sum(x[2] for x in L)/len(L):.1f}, "
          f"p95 extra {sorted(x[2] for x in L)[int(.95*len(L))]}")
