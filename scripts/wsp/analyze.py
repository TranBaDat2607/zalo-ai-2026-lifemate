"""Weight-space edit search: generation arms against the unedited arm, on the fixed D1 / D2 halves (log.md, 11 Nov).

    python scripts/wsp/analyze.py outputs/wsp-alpha0 --base a1.0 [--tf outputs/wsp-tf0/tf]

Each subfolder of RUN with detail.*.jsonl is an arm. Per arm: D1, D2, all-386 and held / random half scores, paired
gains against --base with a bootstrap 95 % range, repairs, asks, abnormal stops, and the rule verdicts (2 and 3).
With --tf, the teacher-forced probe's per-program mean NLL / greedy-exact rate on D1 and D2, and the calibration rule
(1) against the generation arms whose name is a<alpha> (program {"all": alpha}).
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data  # noqa: E402

SPLITS = json.loads((REPO / "scripts" / "wsp" / "wsp_split.json").read_text())
D1 = D2 = HELD = RANDOM = set()


def use_split(name: str) -> None:
    """fold0: the 386 (held-out apps + random); fold1: folds_dpo fold 1 (387 random samples, no held-out apps)."""
    global D1, D2, HELD, RANDOM
    D1, D2 = set(SPLITS[name]["discover"]), set(SPLITS[name]["confirm"])
    if name == "fold0":
        HELD, RANDOM = data.id_set("cvheld:0"), data.id_set("cvrandom:0")
    else:
        HELD, RANDOM = set(), D1 | D2


def load_arm(folder: Path) -> dict:
    rows = {}
    for f in sorted(folder.glob("detail.*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def paired(ids, a, b, rounds=4000, seed=0):
    ids = sorted(i for i in ids if i in a and i in b)
    if not ids:
        return None
    d = [100 * (a[i]["score"] - b[i]["score"]) for i in ids]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(rounds))
    return {"n": len(ids), "gain": round(sum(d) / len(d), 2), "lo": round(means[int(0.025 * rounds)], 2),
            "hi": round(means[int(0.975 * rounds)], 2), "changed": sum(x != 0 for x in d)}


def mean(ids, a):
    ids = [i for i in ids if i in a]
    return round(100 * sum(a[i]["score"] for i in ids) / max(len(ids), 1), 2)


def counts(ids, a):
    ids = [i for i in ids if i in a]
    return {"repairs": sum(a[i].get("repairs", 0) for i in ids), "asked": sum(a[i].get("stop") == "asked" for i in ids),
            "abnormal": sum(a[i].get("stop") not in ("done", "asked") for i in ids)}


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda k: v[k])
        r = [0.0] * len(v)
        for pos, k in enumerate(order):
            r[k] = pos
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--base", default="a1.0")
    ap.add_argument("--tf", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="fold1", choices=("fold0", "fold1"))
    args = ap.parse_args()
    use_split(args.split)
    run = Path(args.run)
    arms = {d.name: load_arm(d) for d in sorted(run.iterdir()) if d.is_dir() and list(d.glob("detail.*.jsonl"))}
    base = arms[args.base]
    report = {}
    print(f"{'arm':8} {'n':>4} {'D1':>6} {'D2':>6} {'all':>6} {'held':>6} {'rand':>6} | gain D1 [95%]        D2      all [95%]"
          "            held   rand | rep ask abn")
    for name, a in arms.items():
        g = {k: paired(ids, a, base) for k, ids in (("D1", D1), ("D2", D2), ("all", D1 | D2), ("held", HELD), ("random", RANDOM))}
        c, cb = counts(D1 | D2, a), counts(D1 | D2, base)
        rec = {"n": len(a), "D1": mean(D1, a), "D2": mean(D2, a), "all": mean(D1 | D2, a), "held": mean(HELD, a),
               "random": mean(RANDOM, a), "gains": g, **c}
        if name != args.base and g["D1"] and g["all"]:
            rec["candidate (rule 2: D1 gain >= +1.0)"] = g["D1"]["gain"] >= 1.0
            rec["confirmed (rule 3)"] = bool(g["D1"]["gain"] >= 1.0 and g["D2"]["gain"] >= 0 and g["all"]["gain"] >= 1.0
                                             and (g["held"] is None or g["held"]["gain"] >= -1.0) and g["random"]["gain"] >= -1.0
                                             and c["repairs"] - cb["repairs"] <= 3 and c["asked"] - cb["asked"] <= 8)
        report[name] = rec
        gs = lambda k: f"{g[k]['gain']:+6.2f}" if g[k] else "   n/a"  # noqa: E731
        rng = lambda k: f"[{g[k]['lo']:+.1f},{g[k]['hi']:+.1f}]" if g[k] else ""  # noqa: E731
        print(f"{name:8} {len(a):4} {rec['D1']:6.2f} {rec['D2']:6.2f} {rec['all']:6.2f} {rec['held']:6.2f} {rec['random']:6.2f} | "
              f"{gs('D1')} {rng('D1'):14} {gs('D2')} {gs('all')} {rng('all'):14} {gs('held')} {gs('random')} | "
              f"{c['repairs']:3} {c['asked']:3} {c['abnormal']:3}"
              + ("  CONFIRMED" if rec.get("confirmed (rule 3)") else ("  candidate" if rec.get("candidate (rule 2: D1 gain >= +1.0)") else "")))
    if args.tf:
        rows = []
        for f in sorted(Path(args.tf).glob("tf.*.jsonl")):
            rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        progs = {}
        for r in rows:
            progs.setdefault(r["program"], {})[r["id"]] = r
        base_key = json.dumps({"all": 1.0}, sort_keys=True)
        print("\nteacher-forced probe (per-sample mean NLL per token; exact = model picks every gold token)")
        tf = {}
        for key, by in progs.items():
            out = {}
            for part, ids in (("D1", D1), ("D2", D2)):
                ids = [i for i in ids if i in by]
                out[part] = {"n": len(ids), "nll_tok": round(sum(by[i]["nll"] for i in ids) / max(sum(by[i]["n"] for i in ids), 1), 5),
                             "exact": round(100 * sum(by[i]["exact"] for i in ids) / max(len(ids), 1), 2),
                             "hit": round(100 * sum(by[i]["hit"] for i in ids) / max(sum(by[i]["n"] for i in ids), 1), 3)}
                if base_key in progs:
                    common = [i for i in ids if i in progs[base_key]]
                    out[part]["better_nll"] = sum(by[i]["nll"] < progs[base_key][i]["nll"] - 1e-6 for i in common)
                    out[part]["worse_nll"] = sum(by[i]["nll"] > progs[base_key][i]["nll"] + 1e-6 for i in common)
            tf[key] = out
            print(f"{key:36} D1 nll/tok {out['D1']['nll_tok']:.5f} exact {out['D1']['exact']:6.2f} hit {out['D1']['hit']:7.3f} | "
                  f"D2 nll/tok {out['D2']['nll_tok']:.5f} exact {out['D2']['exact']:6.2f}"
                  + (f" | D1 better/worse {out['D1'].get('better_nll')}/{out['D1'].get('worse_nll')}" if base_key in progs else ""))
        report["_tf"] = tf
        pairs = []
        for name in arms:
            if name.startswith("a"):
                key = json.dumps({"all": float(name[1:])}, sort_keys=True)
                if key in tf:
                    pairs.append((float(name[1:]), report[name]["D1"], tf[key]["D1"]["nll_tok"]))
        if len(pairs) >= 4:
            pairs.sort()
            rho = spearman([p[1] for p in pairs], [-p[2] for p in pairs])
            best_gen = max(pairs, key=lambda p: p[1])[0]
            best_nll = min(pairs, key=lambda p: p[2])[0]
            grid = [p[0] for p in pairs]
            near = abs(grid.index(best_gen) - grid.index(best_nll)) <= 1
            report["_calibration"] = {"pairs": pairs, "spearman_gen_vs_negnll": round(rho, 3), "best_gen": best_gen,
                                      "best_nll": best_nll, "rule1_pass": rho >= 0.6 and near}
            print(f"\ncalibration on D1: Spearman(gen score, -NLL) = {rho:.3f}; best by generation α={best_gen}, by NLL α={best_nll};"
                  f" rule 1 {'PASS' if rho >= 0.6 and near else 'FAIL'}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
