"""A learned chooser (conditional logistic regression) over the five speech copies' readings, 16 Nov 2026.

Where the five single readings disagree with the averaged one (the spans of the readings view), pick the option a
softmax-over-options logistic regression scores highest, instead of always the averaged reading. Fitted on the
out-of-fold train readings outside the fixed validation split, judged on the latest requests of the 386 validation
samples (word errors after asr_eval.normal_tokens), and, only when it passes --gate, applied to the latest request of
each public sample: writes {out}/stacked_public_test.json (the averaged reading with the chosen options put in) and
{out}/changed_ids.json (samples whose latest request changed), for infer --transcripts view:... --ids-file ...

    python -m lifemate.stack_view --out OUT --train auto:avg5_train.json,auto:medium_train.json,... \
        --public auto:avg5_public_test.json,auto:medium_public_test.json,... --gate 0.5
"""
import argparse
import collections
import difflib
import json
import math
from pathlib import Path

import numpy as np

from . import data
from .infer import load_transcripts
from .view import _TRAIL, _units, _bare, MIN_RATIO

NUMBER_WORDS = frozenset("không một mốt hai ba bốn năm lăm tư sáu bảy tám chín mười mươi trăm nghìn ngàn triệu tỷ "
                         "rưỡi giờ phút tiếng ngày tháng tuần".split())


def spans(main: str, others: list) -> tuple:
    """As view.spans_of, but each option keeps which readings gave it: (main surface words, main units,
    [(first word, last word, a, b, [(option text, option tokens, set of reading indexes)])]); option 0 is the main one."""
    words, units = _units(main)
    seq = [t for t, _ in units]
    n = len(seq)
    if not n:
        return words, units, []
    readings, ranges = [], []
    for r, text in enumerate(others):
        if not text or not text.strip():
            continue
        o_words, o_units = _units(text)
        o_seq = [t for t, _ in o_units]
        matcher = difflib.SequenceMatcher(a=seq, b=o_seq, autojunk=False)
        if not o_seq or matcher.ratio() < MIN_RATIO:
            continue
        match = {}
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                match.update({i1 + d: j1 + d for d in range(i2 - i1)})
                continue
            if i1 == i2 or j1 == j2:
                i1, i2 = (i1 - 1, i2) if i1 > 0 else (i1, i2 + 1)
            ranges.append((i1, min(i2, n)))
        readings.append((r, o_words, o_units, match))
    word_of = [k for _, k in units]
    first, last = {}, {}
    for idx, k in enumerate(word_of):
        first.setdefault(k, idx)
        last[k] = idx
    merged = []
    for a, b in sorted(ranges):
        a, b = first[word_of[a]], last[word_of[b - 1]] + 1
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        main_key = " ".join(seq[a:b])
        givers, texts = collections.defaultdict(set), {}
        for r, o_words, o_units, match in readings:
            s = match[a - 1] + 1 if a > 0 else 0
            e = match[b] if b < n else len(o_units)
            key = " ".join(t for t, _ in o_units[s:e])
            givers[key].add(r)
            if key != main_key and key not in texts:
                texts[key] = _bare(o_words[o_units[s][1]:o_units[e - 1][1] + 1]) if s < e else ""
        options = [(_bare(words[word_of[a]:word_of[b - 1] + 1]), main_key.split(), givers.get(main_key, set()))]
        for key in sorted((k for k in givers if k != main_key), key=lambda k: -len(givers[k])):
            options.append((texts[key], key.split(), givers[key]))
        if len(options) > 1:
            out.append((word_of[a], word_of[b - 1], a, b, options))
    return words, units, out


def tokens_with(units: list, sp: list, pick: list) -> list:
    seq, out, last = [t for t, _ in units], [], 0
    for (_, _, a, b, options), p in zip(sp, pick):
        out += seq[last:a] + options[p][1]
        last = b
    return out + seq[last:]


def render(words: list, sp: list, pick: list) -> str:
    out, k, at = [], 0, {f: (l, opts[p][0]) for (f, l, _, _, opts), p in zip(sp, pick) if p}
    while k < len(words):
        if k in at:
            l, text = at[k]
            trail = _TRAIL.search(words[l])
            if text:
                out.append(text + (trail.group() if trail else ""))
            k = l + 1
        else:
            out.append(words[k])
            k += 1
    return " ".join(out)


def errors(a: list, b: list) -> int:
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in
               difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op != "equal")


def features(sp_entry, n_readings: int, freq: collections.Counter) -> np.ndarray:
    _, _, a, b, options = sp_entry
    main_len = len(options[0][1])
    rows = []
    for k, (text, toks, givers) in enumerate(options):
        logf = [math.log1p(freq[t]) for t in toks] or [0.0]
        rows.append([1.0 if k == 0 else 0.0, len(givers), *[1.0 if r in givers else 0.0 for r in range(n_readings)],
                     len(toks) - main_len, 1.0 if not toks else 0.0, min(logf), sum(logf) / len(logf),
                     sum(t not in freq for t in toks), sum(t in NUMBER_WORDS for t in toks),
                     1.0 if k == 0 and len(givers) == 0 else 0.0, len(options)])
    return np.array(rows, dtype=np.float64)


def fit(groups: list, l2: float = 1e-2, steps: int = 3000, lr: float = 0.05) -> tuple:
    """Softmax over each span's options with a linear score (conditional logit), soft targets, Adam."""
    X = np.concatenate([g[0] for g in groups])
    Y = np.concatenate([g[1] for g in groups])
    starts = np.cumsum([0] + [len(g[1]) for g in groups[:-1]])
    group = np.repeat(np.arange(len(groups)), [len(g[1]) for g in groups])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    w = np.zeros(X.shape[1])
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t in range(1, steps + 1):
        s = X @ w
        s = np.exp(s - np.maximum.reduceat(s, starts)[group])
        p = s / np.add.reduceat(s, starts)[group]
        grad = l2 * w + X.T @ (p - Y) / len(groups)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w, mu, sd


def choose(sp: list, n_readings: int, freq, model) -> list:
    w, mu, sd = model
    return [int(np.argmax(((features(e, n_readings, freq) - mu) / sd) @ w)) for e in sp]


def save_model(path, model, n_readings: int, freq) -> None:
    w, mu, sd = model
    Path(path).write_text(json.dumps({"weights": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "n_readings": n_readings,
                                      "freq": dict(freq)}, ensure_ascii=False), encoding="utf-8")


def load_model(path) -> dict:
    if str(path).startswith("auto:"):  # a file in an attached Kaggle dataset
        import os
        name = path[len("auto:"):]
        path = next(os.path.join(d, name) for d, _, files in os.walk("/kaggle/input") if name in files)
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"model": (np.array(m["weights"]), np.array(m["mu"]), np.array(m["sd"])), "n_readings": m["n_readings"],
            "freq": collections.Counter(m["freq"])}


def restack(main: str, others: list, stack: dict) -> str:
    """The main reading with the stacker's chosen options put in (unchanged where it keeps the main ones)."""
    if len(others) != stack["n_readings"]:
        return main
    words, units, sp = spans(main, others)
    pick = choose(sp, stack["n_readings"], stack["freq"], stack["model"])
    return render(words, sp, pick) if any(pick) else main


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", required=True, help="main reading first, then the single readings (train, out of fold)")
    ap.add_argument("--public", default=None, help="the same readings of the public test, in the same order")
    ap.add_argument("--gate", type=float, default=0.5, help="needed drop of validation word error rate, in points")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only this many train samples")
    ap.add_argument("--save", default=None, help="write the fitted chooser here (solve.py --stack-model)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    main_t, *others_t = [load_transcripts(p) for p in args.train.split(",")]
    n_read = len(others_t)
    items = data.load_split("train", data.find_root("train"))
    if args.limit:
        items = items[:args.limit]
    val = data.id_set("splits")
    clips = []  # (sample id, is latest, audio, gold)
    for it in items:
        clips.append((it["id"], True, it["audio"], it.get("audio_text")))
        for h in it.get("history") or []:
            if h.get("role") == "user" and h.get("audio") and h.get("audio_text"):
                clips.append((it["id"], False, h["audio"], h["audio_text"]))
    freq = collections.Counter()
    for sid, _, _, gold in clips:
        if sid not in val and gold:
            freq.update(t for t, _ in _units(gold)[1])

    groups, evals = [], []
    for sid, latest, audio, gold in clips:
        if audio not in main_t or not gold:
            continue
        words, units, sp = spans(main_t[audio], [o.get(audio, "") for o in others_t])
        g = [t for t, _ in _units(gold)[1]]
        if sid in val:
            if latest:
                evals.append((words, units, sp, g))
            continue
        for s, e in enumerate(sp):
            base = [0] * len(sp)
            errs = []
            for p in range(len(e[4])):
                base[s] = p
                errs.append(errors(tokens_with(units, sp, base), g))
            best = min(errs)
            if all(x == best for x in errs):
                continue  # no option is better than another: nothing to learn
            y = np.array([1.0 if x == best else 0.0 for x in errs])
            groups.append((features(e, n_read, freq), y / y.sum()))
    print(f"train spans {len(groups)}, validation latest clips {len(evals)}", flush=True)
    model = fit(groups)

    report = collections.Counter()
    for words, units, sp, g in evals:
        report["words"] += len(g)
        report["clips"] += 1
        pick = choose(sp, n_read, freq, model)
        stacked = [t for t, _ in _units(render(words, sp, pick))[1]]
        for name, toks in (("main", [t for t, _ in units]), ("stacked", stacked)):
            e = errors(toks, g)
            report[name + " errors"] += e
            report[name + " exact"] += e == 0
        report["changed clips"] += any(pick)
    wer = {k: 100 * report[k + " errors"] / max(report["words"], 1) for k in ("main", "stacked")}
    result = {"validation": dict(report), "wer": wer, "drop": wer["main"] - wer["stacked"], "gate": args.gate,
              "weights": model[0].tolist()}
    result["pass"] = result["drop"] >= args.gate
    print(json.dumps(result, indent=1), flush=True)
    (out / "stack_gate.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if args.save:
        save_model(args.save, model, n_read, freq)
    if not result["pass"]:
        raise SystemExit(f"gate failed: WER drop {result['drop']:.2f} < {args.gate}")
    if not args.public:
        return

    main_p, *others_p = [load_transcripts(p) for p in args.public.split(",")]
    stacked, changed = dict(main_p), []
    for it in data.load_split("public_test", data.find_root("public_test")):
        audio = it["audio"]
        if audio not in main_p:
            continue
        words, units, sp = spans(main_p[audio], [o.get(audio, "") for o in others_p])
        pick = choose(sp, n_read, freq, model)
        if any(pick):
            stacked[audio] = render(words, sp, pick)
            changed.append(it["id"])
    (out / "stacked_public_test.json").write_text(json.dumps(stacked, ensure_ascii=False, indent=0), encoding="utf-8")
    (out / "changed_ids.json").write_text(json.dumps(changed), encoding="utf-8")
    print(f"public: {len(changed)} samples changed", flush=True)


if __name__ == "__main__":
    main()
