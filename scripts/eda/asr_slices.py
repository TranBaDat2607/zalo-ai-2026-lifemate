"""Speech-model errors (five-copy out-of-fold transcripts) by batch, dialect, audio quality and word position."""
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import REPO, gen_by_id, load  # noqa: E402
from lifemate.asr_eval import normal_tokens  # noqa: E402

SOUTH = {"nha", "nghen", "hông", "hổng", "giùm", "dùm", "bữa", "tui", "ổng", "bả", "nè", "hen", "nhiêu", "xài", "mấy", "đặng", "vô", "ghé", "lẹ", "nghe"}
CENTRAL = {"mô", "ni", "rứa", "răng", "chi", "nớ", "tê", "hỉ", "bựa", "mi", "tau", "hè", "nờ", "đọ", "rứa"}
NORTH = {"nhé", "ấy", "nhỉ", "thế", "đấy", "vâng", "nhá", "chứ", "bảo", "đâu"}
STRONG_S = {"nghen", "hông", "hổng", "tui", "ổng", "bả", "nè", "hen", "giùm", "dùm", "bữa", "lẹ"}
STRONG_C = {"mô", "ni", "rứa", "răng", "nớ", "tê", "hỉ", "bựa", "nờ"}
STRONG_N = {"nhé", "nhỉ", "đấy", "nhá"}


def region(text):
    w = set(re.findall(r"[^\W_]+", text.lower()))
    s, c, n = len(w & STRONG_S), len(w & STRONG_C), len(w & STRONG_N)
    if c and c >= s:
        return "Central"
    if s > n:
        return "South"
    if n > s:
        return "North"
    return "unmarked"


def align(ref, hyp):
    """Levenshtein alignment; returns a list of ops per reference token: 'ok', 'sub', 'del' (+ insertions count)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j, ins = [], n, m, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append("ok" if ref[i - 1] == hyp[j - 1] else "sub"); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append("del"); i -= 1
        else:
            ins += 1; j -= 1
    return ops[::-1], ins


if __name__ == "__main__":
    hyp = json.load(open(sys.argv[1], encoding="utf-8"))
    feats = {r["path"].split("/", 1)[1]: r for r in csv.DictReader(open(sys.argv[2], encoding="utf-8")) if r["path"].startswith("train/")}
    rows = []
    for s in load("train"):
        turns = [(s["audio"], s["audio_text"], True)] + [(t["audio"], t["audio_text"], False) for t in s["history"] if t["role"] == "user"]
        for a, ref, final in turns:
            if a not in hyp:
                continue
            r, h = normal_tokens(ref), normal_tokens(hyp[a])
            ops, ins = align(r, h)
            f = feats.get(a, {})
            rows.append(dict(sid=s["id"], gen=gen_by_id(s["id"]), final=final, region=region(ref), n=len(r),
                             err=sum(o != "ok" for o in ops) + ins, last_bad=bool(ops) and ops[-1] != "ok",
                             first_bad=bool(ops) and ops[0] != "ok", mid_bad=sum(o != "ok" for o in ops[1:-1]), mid_n=max(0, len(ops) - 2),
                             snr=float(f.get("snr_db", "nan")), dur=float(f.get("dur", "nan")), trail=float(f.get("trail", "nan"))))

    def wer(rs):
        return 100 * sum(r["err"] for r in rs) / max(1, sum(r["n"] for r in rs))

    def show(title, key, order=None):
        g = collections.defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        print(title)
        for k in (order or sorted(g)):
            if k in g:
                print(f"   {str(k):12s} turns={len(g[k]):5d} WER={wer(g[k]):5.2f}%  exact={100*sum(r['err']==0 for r in g[k])/len(g[k]):4.1f}%")

    print(f"all turns {len(rows)}  WER {wer(rows):.2f}%")
    show("by batch", lambda r: r["gen"])
    show("by turn", lambda r: "final" if r["final"] else "history")
    show("by dialect proxy (strong marker words in the reference)", lambda r: r["region"], ["North", "South", "Central", "unmarked"])
    show("by SNR (dB)", lambda r: "<20" if r["snr"] < 20 else ("20-35" if r["snr"] < 35 else ("35-50" if r["snr"] < 50 else ">=50")), ["<20", "20-35", "35-50", ">=50"])
    show("by trailing silence", lambda r: "cut (<=0.04s)" if r["trail"] <= 0.04 else "has tail")
    show("by length (words)", lambda r: "<=6" if r["n"] <= 6 else ("7-12" if r["n"] <= 12 else ("13-20" if r["n"] <= 20 else ">20")), ["<=6", "7-12", "13-20", ">20"])
    tot = len(rows)
    print(f"position error rates: first word {100*sum(r['first_bad'] for r in rows)/tot:.1f}%, last word {100*sum(r['last_bad'] for r in rows)/tot:.1f}%, middle words {100*sum(r['mid_bad'] for r in rows)/max(1,sum(r['mid_n'] for r in rows)):.1f}%")
    cut = [r for r in rows if r["trail"] <= 0.04]; tail = [r for r in rows if r["trail"] > 0.04]
    print(f"last-word error: trimmed clips {100*sum(r['last_bad'] for r in cut)/len(cut):.1f}% vs clips with a tail {100*sum(r['last_bad'] for r in tail)/len(tail):.1f}%")
    reg = collections.Counter((r["gen"], r["region"]) for r in rows if r["final"])
    print("dialect proxy by batch (final turns):", dict(sorted(reg.items())))
