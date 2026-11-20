"""How reliable is a word when all five readings agree with the avg5 text, versus when they disagree?
Per clip: avg5 words aligned to gold (correct = an 'equal' opcode); a word is 'agreed' when no reading differs from avg5
around it (outside every alternative span of readings_view.view)."""
import collections as C, difflib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from keyword_pool import toks, value_tokens, NUMW, TIME, items, readings  # noqa: E402
from readings_view import view, five  # noqa: E402

st = C.Counter()
for sid, it in items.items():
    calls = it["label"]["tool_calls"]
    argw = set().union(*[value_tokens(c.get("arguments") or {}) for c in calls]) if calls else set()
    for audio, gold in [(it["audio"], it["audio_text"])] + [(h["audio"], h["audio_text"]) for h in it.get("history") or []
                                                           if h.get("role") == "user" and h.get("audio")]:
        if audio not in readings["avg5"]:
            continue
        main, G = toks(readings["avg5"][audio]), toks(gold)
        correct = [False] * len(main)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=main, b=G, autojunk=False).get_opcodes():
            if op == "equal":
                for i in range(i1, i2):
                    correct[i] = True
        disputed = [False] * len(main)
        for i1, i2, _ in view(main, [toks(readings[k][audio]) for k in five if audio in readings[k]]):
            for i in range(max(0, i1 - (i1 == i2)), max(i2, i1 + 1)):
                if i < len(main):
                    disputed[i] = True
        for i, w in enumerate(main):
            key = w in argw or w in NUMW or w in TIME
            tag = ("disputed" if disputed[i] else "agreed") + (" key" if key else "")
            st[tag + " words"] += 1
            st[tag + " correct"] += correct[i]
        # gold words missing from avg5 (deletions) are not counted here; substitutions are
for tag in ["agreed", "disputed", "agreed key", "disputed key"]:
    n, c = st[tag + " words"], st[tag + " correct"]
    print(f"{tag:<14} words {n:>6}  correct {100 * c / n:5.1f} %  wrong {n - c}")
