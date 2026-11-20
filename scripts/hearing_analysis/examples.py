import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from keyword_pool import toks, items, readings, run  # noqa: E402
from readings_view import view, five  # noqa: E402
from first_error import restated  # noqa: E402

def render(audio):
    main = toks(readings["avg5"][audio])
    v = view(main, [toks(readings[k][audio]) for k in five])
    out, i = [], 0
    for i1, i2, alts in v:
        out += main[i:i1]
        here = " ".join(main[i1:i2])
        opts = [here] + sorted(a for a in alts if a != here)
        out.append("{" + "|".join(o if o else "∅" for o in opts) + "}")
        i = i2
    return " ".join(out + main[i:])

for sid in sys.argv[1:]:
    it, r = items[sid], run[sid]
    print(sid, "score", r["score"])
    print("  said    :", it["audio_text"])
    print("  view    :", render(it["audio"]))
    print("  restated:", restated(r))
