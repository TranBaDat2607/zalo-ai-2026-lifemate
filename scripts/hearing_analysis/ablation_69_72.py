"""How much did (1) the speech-model changes and (2) the agent's restating step (the Qwen3.5 rewrite of the request,
`--denoise`) change the quality of the latest request in the 69.72 pipeline (0019_view-full2)?  CPU only, saved runs.

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python3 scripts/hearing_analysis/ablation_69_72.py

Part A  speech only (no agent): exp7 single model vs five-copy average (avg5) vs the readings view (avg5 + braces),
        latest request, all 1,159 train samples (out-of-fold readings) and the 386 validation samples.
Part B  restating, validation (gold text known): heard text H vs restated text R vs said text G, per word of G:
        fixed (H wrong, R right), broken (H right, R wrong), still wrong, plus words R adds. Runs:
          soup+exp7   w3-soup/soup_eval      (same LLM as the next row: only the speech model differs)
          soup+avg5   harm-check5/avg5_eval
          0012+avg5   notes-aux/aux_eval     (side-task model, the 69.72 recipe minus the view)
          0012+gold   stage0-latest/oracle_latest (perfect latest text in: every change R makes is an error)
Part C  restating, public (no gold): what the 69.72 model (view) and aux-full4 (same recipe, plain avg5 text) do with
        words all six readings agree on and with disputed spans.
Part D  score pairs already measured (paired bootstrap on validation; public scores as uploaded).
"""
import collections as C
import difflib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "scripts" / "notes"))
from build_pairs import FILLERS  # noqa: E402
from compare_notes import SPLITS, load  # noqa: E402
from first_error import restated  # noqa: E402
from lifemate import data, view  # noqa: E402
from lifemate.asr_eval import edit_distance, normal_tokens  # noqa: E402

TIME = set("sáng trưa chiều tối đêm khuya mai nay mốt qua hôm giờ phút rưỡi kém thứ tuần tháng năm ngày sau trước "
           "tới hai ba tư bốn năm sáu bảy tám chín mười mươi chủ nhật".split())
NUMW = set("không một mốt hai ba bốn tư năm lăm sáu bảy bẩy tám chín mười mươi trăm nghìn ngàn triệu tỷ linh lẻ rưỡi".split())
TRAIN = {"avg5": "outputs/asr-avg5/avg5_train.json", "exp7": "outputs/exp7/asr_medium/medium_train.json",
         "m10": "outputs/w1-asr/asr_m10/medium10_train.json", "B": "outputs/b-synth/b/b_train.json",
         "s2027": "outputs/asr-s2027/s2027_train.json", "s2028": "outputs/asr-s2028/s2028_train.json"}
PUBLIC = {"avg5": "outputs/final1/avg5_asr/avg5_public_test.json", "exp7": "outputs/exp7/asr_medium/medium_public_test.json",
          "m10": "outputs/w1-asr/asr_m10/medium10_public_test.json", "B": "outputs/b-synth-s2/b/public_test.json",
          "s2027": "outputs/asr-s2027/s2027/public_test.json", "s2028": "outputs/asr-s2028/s2028/public_test.json"}
FIVE = ["exp7", "m10", "B", "s2027", "s2028"]
RUNS = {"soup+exp7": "outputs/w3-soup/soup_eval", "soup+avg5": "outputs/harm-check5/avg5_eval",
        "0012+avg5": "outputs/notes-aux/aux_eval", "0012+gold": "outputs/stage0-latest/oracle_latest"}
rng = random.Random(2026)
OUT = {}


def toks(text):
    return [w for w in normal_tokens(text or "") if w not in FILLERS]


def value_tokens(v):
    if isinstance(v, bool) or v is None:
        return set()
    if isinstance(v, (int, float)):
        return set(normal_tokens(str(int(v)) if float(v).is_integer() else str(v)))
    if isinstance(v, str):
        return set(normal_tokens(v))
    if isinstance(v, list):
        return set().union(set(), *[value_tokens(x) for x in v])
    if isinstance(v, dict):
        return set().union(set(), *[value_tokens(x) for x in v.values()])
    return set()


def right_positions(G, X):
    """Positions of G matched exactly in X, and the number of X words matched to nothing in G (additions)."""
    sm = difflib.SequenceMatcher(a=G, b=X, autojunk=False)
    ok = set()
    for i, j, n in sm.get_matching_blocks():
        ok.update(range(i, i + n))
    return ok, len(X) - len(ok)


def boot(pairs, rounds=4000):
    """pairs = [(errors_a, errors_b, words)] per request -> (diff in WER points, low, high) of b - a."""
    n = len(pairs)
    tot = lambda s: 100 * (sum(p[1] for p in s) - sum(p[0] for p in s)) / max(1, sum(p[2] for p in s))
    d = sorted(tot(rng.choices(pairs, k=n)) for _ in range(rounds))
    return round(tot(pairs), 2), round(d[int(.025 * rounds)], 2), round(d[int(.975 * rounds) - 1], 2)


def keyset(it):
    w = set()
    for c in (it.get("label") or {}).get("tool_calls") or []:
        w |= value_tokens(c.get("arguments") or {})
    return w


root = data.find_root("train")
items = {s["id"]: s for s in data.load_split("train", root)}
val = list(SPLITS["random"]) + list(SPLITS["apps"])
rd = {k: json.load(open(REPO / p)) for k, p in TRAIN.items()}


def oracle_view(main, others, G):
    """Best text the agent could write by picking one option per brace (greedy per span), and the view's spans."""
    words, kept = view.choose(main, others)
    choice = [0] * len(kept)

    def text(ch):
        out, k, at = [], 0, {f: (l, o) for f, l, o in kept}
        idx = {f: n for n, (f, _, _) in enumerate(kept)}
        while k < len(words):
            if k in at:
                out.append(at[k][1][ch[idx[k]]]); k = at[k][0] + 1
            else:
                out.append(words[k]); k += 1
        return " ".join(out)
    for _ in range(2):
        for s, (_, _, opts) in enumerate(kept):
            best = min(range(len(opts)), key=lambda o: edit_distance(G, toks(text(choice[:s] + [o] + choice[s + 1:]))))
            choice[s] = best
    return toks(text(choice)), kept


# ---------- Part A: speech only ----------
def part_a(ids, label):
    rows = []
    for sid in ids:
        it = items[sid]; G = toks(it["audio_text"]); a = it["audio"]
        if a not in rd["avg5"] or a not in rd["exp7"]:
            continue
        kw = keyset(it) | NUMW | TIME
        ke = {}
        e = {}
        for name in ("exp7", "avg5"):
            X = toks(rd[name][a]); e[name] = edit_distance(G, X)
            ok, _ = right_positions(G, X)
            ke[name] = sum(1 for i, w in enumerate(G) if w in kw and i not in ok)
        Xo, kept = oracle_view(rd["avg5"][a], [rd[m].get(a, "") for m in FIVE], G)
        e["view_best"] = edit_distance(G, Xo)
        ok, _ = right_positions(G, Xo)
        ke["view_best"] = sum(1 for i, w in enumerate(G) if w in kw and i not in ok)
        nk = sum(1 for w in G if w in kw)
        rows.append(dict(id=sid, n=len(G), nk=nk, e=e, ke=ke, braces=len(kept),
                         exact={k: e[k] == 0 for k in e}))
    W = sum(r["n"] for r in rows); K = sum(r["nk"] for r in rows)
    res = {"requests": len(rows), "words": W, "key words": K,
           "share with braces": round(sum(r["braces"] > 0 for r in rows) / len(rows), 3)}
    for k in ("exp7", "avg5", "view_best"):
        res[k] = {"WER %": round(100 * sum(r["e"][k] for r in rows) / W, 2),
                  "key words wrong %": round(100 * sum(r["ke"][k] for r in rows) / K, 2),
                  "exact requests %": round(100 * sum(r["exact"][k] for r in rows) / len(rows), 1)}
    res["avg5 - exp7 WER (95%)"] = boot([(r["e"]["exp7"], r["e"]["avg5"], r["n"]) for r in rows])
    res["avg5 - exp7 key wrong (95%)"] = boot([(r["ke"]["exp7"], r["ke"]["avg5"], r["nk"]) for r in rows])
    res["best brace pick - avg5 WER (95%)"] = boot([(r["e"]["avg5"], r["e"]["view_best"], r["n"]) for r in rows])
    res["best brace pick - avg5 key wrong (95%)"] = boot([(r["ke"]["avg5"], r["ke"]["view_best"], r["nk"]) for r in rows])
    OUT["A " + label] = res


part_a(sorted(items), "all train samples (latest request, out-of-fold)")
part_a(val, "validation 386")


# ---------- Part B: restating on validation ----------
def part_b(name, path):
    run = load(str(REPO / path))
    st = C.Counter(); per = []
    for sid in val:
        r = run[sid]; it = items[sid]
        G, H, R = toks(it["audio_text"]), toks(r["request"]), toks(restated(r))
        kw = keyset(it) | NUMW | TIME
        okH, addH = right_positions(G, H); okR, addR = right_positions(G, R)
        row = C.Counter()
        for i, w in enumerate(G):
            tag = ("ok" if i in okH else "bad") + ">" + ("ok" if i in okR else "bad")
            row[tag] += 1
            if w in kw:
                row["key " + tag] += 1
        row["added H"], row["added R"] = addH, addR
        row["wer H"], row["wer R"], row["n"] = edit_distance(G, H), edit_distance(G, R), len(G)
        row["copied"] = int(H == R)
        st.update(row)
        per.append((sid, row, r["score"]))
    W = st["n"]
    keyn = sum(st["key " + t] for t in ("ok>ok", "ok>bad", "bad>ok", "bad>bad"))
    res = {
        "score": round(100 * sum(s for _, _, s in per) / len(per), 2),
        "restatement = heard text (requests)": st["copied"],
        "heard WER %": round(100 * st["wer H"] / W, 2), "restated WER %": round(100 * st["wer R"] / W, 2),
        "restated - heard WER (95%)": boot([(r["wer H"], r["wer R"], r["n"]) for _, r, _ in per]),
        "words fixed": st["bad>ok"], "words broken (were right)": st["ok>bad"], "words still wrong": st["bad>bad"],
        "words added: heard / restated": (st["added H"], st["added R"]),
        "key words fixed": st["key bad>ok"], "key words broken": st["key ok>bad"],
        "key words still wrong": st["key bad>bad"], "key words": keyn,
        "key wrong % heard -> restated": (round(100 * (st["key bad>ok"] + st["key bad>bad"]) / keyn, 2),
                                          round(100 * (st["key ok>bad"] + st["key bad>bad"]) / keyn, 2)),
    }
    br = [(s, sc) for s, r, sc in per if r["key ok>bad"] and not r["key bad>ok"]]
    fx = [(s, sc) for s, r, sc in per if r["key bad>ok"] and not r["key ok>bad"]]
    res["requests: restating only broke key words (n, mean score)"] = (len(br), round(100 * sum(x for _, x in br) / max(1, len(br)), 1))
    res["requests: restating only fixed key words (n, mean score)"] = (len(fx), round(100 * sum(x for _, x in fx) / max(1, len(fx)), 1))
    OUT["B " + name] = res
    return {s: (r, sc) for s, r, sc in per}


B = {k: part_b(k, p) for k, p in RUNS.items()}


# ---------- Part C: restating on public, view model vs plain-text model ----------
def pub_rows(path):
    rows = {}
    for f in sorted((REPO / path).glob("detail.*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line); rows[x["id"]] = x
    return rows


pub = {k: json.load(open(REPO / p)) for k, p in PUBLIC.items()}
vrun, arun = pub_rows("outputs/public-view2/public_view2"), pub_rows("outputs/public-aux4/public_aux4")
proot = data.find_root("public_test")
pitems = {s["id"]: s for s in data.load_split("public_test", proot)}
st = C.Counter()
for sid, it in pitems.items():
    a = it["audio"]
    if sid not in vrun or sid not in arun or a not in pub["avg5"]:
        continue
    main = pub["avg5"][a]
    words, units = view._units(main)
    seq = [t for t, _ in units]
    _, kept = view.choose(main, [pub[m].get(a, "") for m in FIVE])
    in_brace = set()
    spans = []
    for f, l, opts in kept:
        tok_idx = [i for i, (_, k) in enumerate(units) if f <= k <= l]
        in_brace.update(tok_idx)
        spans.append((tok_idx, [" ".join(t for w in o.split() for t in normal_tokens(w) if t not in view.IGNORED) for o in opts]))
    st["requests"] += 1; st["braces"] += len(kept)
    for tag, run in (("view", vrun), ("plain", arun)):
        R = [t for t in normal_tokens(restated(run[sid])) if t not in view.IGNORED]
        sm = difflib.SequenceMatcher(a=seq, b=R, autojunk=False)
        match = {}
        for i, j, n in sm.get_matching_blocks():
            match.update({i + d: j + d for d in range(n)})
        agreed = [i for i in range(len(seq)) if i not in in_brace]
        st[tag + " agreed words"] += len(agreed)
        st[tag + " agreed words changed"] += sum(i not in match for i in agreed)
        st[tag + " requests with an agreed word changed"] += any(i not in match for i in agreed)
        ops = sm.get_opcodes()
        for tok_idx, opts in spans:
            if not tok_idx:
                continue
            a0, a1 = tok_idx[0], tok_idx[-1] + 1
            # R's text over the span: between the matched neighbours
            p = a0 - 1
            while p >= 0 and p not in match:
                p -= 1
            q = a1
            while q < len(seq) and q not in match:
                q += 1
            s = match[p] + 1 if p >= 0 else 0
            e = match[q] if q < len(seq) else len(R)
            got = " ".join(R[s:e])
            if got == opts[0]:
                st[tag + " brace: kept main"] += 1
            elif got in opts[1:]:
                st[tag + " brace: took another reading"] += 1
            else:
                st[tag + " brace: wrote something else"] += 1
OUT["C public restatements"] = dict(st)

# ---------- Part D: score pairs from saved runs ----------
def paired(a, b, ids):
    d = [b[i]["score"] - a[i]["score"] for i in ids]
    n = len(d)
    m = sorted(sum(rng.choices(d, k=n)) / n for _ in range(4000))
    return {"a": round(100 * sum(a[i]["score"] for i in ids) / n, 2), "b": round(100 * sum(b[i]["score"] for i in ids) / n, 2),
            "b - a": round(100 * sum(d) / n, 2), "95%": (round(100 * m[100], 2), round(100 * m[3899], 2)),
            "b wins": sum(x > 1e-9 for x in d), "a wins": sum(x < -1e-9 for x in d)}


runs = {k: load(str(REPO / p)) for k, p in RUNS.items()}
fold = {**load(str(REPO / "outputs/exp10/fold_eval")), **load(str(REPO / "outputs/exp10b/fold_eval"))}
fold_dn = load(str(REPO / "outputs/exp9/fold_dn_eval"))
OUT["D validation pairs"] = {
    "speech exp7 -> avg5, same LLM (soup)": paired(runs["soup+exp7"], runs["soup+avg5"], val),
    "restating off -> on (exp9/exp10, early model, exp7 speech)": paired(fold, fold_dn, val),
    "0012 heard latest -> perfect latest (ceiling of hearing)": paired(runs["0012+avg5"], runs["0012+gold"], val),
    "same, random half": paired(runs["0012+avg5"], runs["0012+gold"], SPLITS["random"]),
    "same, unseen-apps half": paired(runs["0012+avg5"], runs["0012+gold"], SPLITS["apps"]),
}
json.dump(OUT, open(Path(__file__).with_name("ablation_69_72.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
