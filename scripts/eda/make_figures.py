"""Draw the EDA figures into figures/eda/ (PNG). Referenced from eda.md.

Counts that come straight from the data (batches, mix, answer types, habits, audio) are computed here. Numbers that
need a validation run or the speech model's out-of-fold transcripts are copied from the outputs of batch_table.py,
lost_points.py, hash_diff.py, post_rules.py and asr_slices.py (7 Nov 2026, run outputs/harm-check5/avg5_eval) and
marked VAL below; rerun those scripts to refresh them.

Usage: python scripts/eda/make_figures.py <audio_feats.csv>   (the CSV comes from audio_feats.py)
"""
import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, gen_by_id, load, num, public_batch  # noqa: E402

OUT = REPO / "figures" / "eda"
BATCH = ["A", "B", "C"]
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}  # categorical slots 1-3, fixed order
MUTED = "#b4b2a9"
INK, INK2, GRID = "#1f1e1c", "#5f5e58", "#e6e4dd"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150,
    "legend.frameon": False,
})


def save(fig, name, title, note=None):
    fig.suptitle(title, x=0.01, y=1.06, ha="left", va="bottom", fontsize=13, fontweight="bold", color=INK)
    if note:
        fig.text(0.01, -0.04, note, ha="left", va="top", fontsize=8, color=INK2)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def label_bars(ax, bars, fmt="{:.1f}", horizontal=False, pad=0.6):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v + pad, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", fontsize=9, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + pad, fmt.format(v), ha="center", fontsize=9, color=INK)


def apps(s):
    return list(s["context"]["installed_apps"])


def main(feats_csv):
    OUT.mkdir(parents=True, exist_ok=True)
    tr, pub = load("train"), load("public_test")
    splits = json.loads((REPO / "src" / "lifemate" / "splits.json").read_text(encoding="utf-8"))
    val_ids = set(splits["random"]) | set(splits["apps"])
    trapps = {a for s in tr for a in apps(s)}

    # 1. batches along the ids: month of the sample's date, train and public
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, rows, bf, name in ((axes[0], tr, lambda s: gen_by_id(s["id"]), "train"),
                               (axes[1], pub, lambda s: public_batch(s["id"]), "public test")):
        for b in BATCH:
            sel = [s for s in rows if bf(s) == b]
            x = [num(s["id"]) for s in sel]
            y = [int(s["context"]["current_time"][5:7]) + rng.uniform(-0.28, 0.28) for s in sel]
            ax.scatter(x, y, s=7, color=COL[b], alpha=0.75, linewidths=0, label=f"batch {b}")
        ax.set_yticks([9, 10, 11, 12], ["Sep", "Oct", "Nov", "Dec"])
        ax.set_title(f"{name}: sample id (x) against the month in the sample's date (y)", fontsize=10, fontweight="normal")
        ax.grid(axis="x", visible=False)
    for x in (425, 917):
        axes[0].axvline(x, color=INK2, lw=1, ls="--")
    for x in (174, 393):
        axes[1].axvline(x, color=INK2, lw=1, ls="--")
    axes[1].set_xlabel("sample id")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=2.5)
    fig.tight_layout()
    save(fig, "01_batches_along_ids.png", "The data was made in three batches, in the same order in train and public",
         "Dashed lines: where date, city and persona pool switch (train 425 / ~917, public 174 / 393).")

    # 2. batch mix per split
    mix = collections.OrderedDict()
    mix["train"] = collections.Counter(gen_by_id(s["id"]) for s in tr)
    mix["validation"] = collections.Counter(gen_by_id(s["id"]) for s in tr if s["id"] in val_ids)
    mix["public test"] = collections.Counter(public_batch(s["id"]) for s in pub)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for i, (name, c) in enumerate(reversed(mix.items())):
        tot, left = sum(c.values()), 0
        for b in BATCH:
            w = 100 * c[b] / tot
            ax.barh(i, w, left=left, color=COL[b], edgecolor="white", linewidth=2, height=0.6, label=f"batch {b}" if i == 0 else None)
            ax.text(left + w / 2, i, f"{b}  {w:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            left += w
        ax.text(101, i, f"n={tot}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(3), list(reversed(mix)))
    ax.set_xlim(0, 110)
    ax.set_xlabel("share of samples (%)")
    ax.grid(False)
    save(fig, "02_batch_mix.png", "Batch shares: validation has fewer A samples than public",
         "Validation's held-out-app half draws only from batches B and C.")

    # 3. batch profile, small multiples
    by = {b: [s for s in tr if gen_by_id(s["id"]) == b] for b in BATCH}
    prof = [
        ("installed apps", [np.mean([len(apps(s)) for s in by[b]]) for b in BATCH], "{:.1f}"),
        ("earlier user turns", [np.mean([sum(h["role"] == "user" for h in s["history"]) for s in by[b]]) for b in BATCH], "{:.2f}"),
        ("gold calls per sample", [np.mean([len(s["label"]["tool_calls"]) for s in by[b]]) for b in BATCH], "{:.2f}"),
        ("speech word errors (%)", [3.52, 8.47, 8.21], "{:.1f}"),  # VAL (asr_slices / batch_table)
        ("validation score now", [79.5, 73.0, 58.4], "{:.1f}"),  # VAL (batch_table)
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
    for ax, (t, vals, fmt) in zip(axes, prof):
        bars = ax.bar(BATCH, vals, color=[COL[b] for b in BATCH], width=0.6)
        ax.set_title(t, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.22)
        label_bars(ax, bars, fmt, pad=max(vals) * 0.03)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "03_batch_profile.png", "Each batch looks different (train)",
         "Speech errors: five-copy model, out of fold. Validation: harm-check5, 386 samples.")

    # 4. answer types by batch
    from hash_diff import DATA  # noqa: F401  (puts the tool library on the path)
    from zaic_tools.apps import TOOLS

    def outcome(calls):
        if not calls:
            return "nothing"
        if calls[-1]["name"] == "sys.ask_missing_info":
            return "ask"
        return "act" if any(TOOLS.get(c["name"], "write") == "write" for c in calls) else "read only"
    kinds = ["act", "ask", "read only", "nothing"]
    kcol = {"act": "#2a78d6", "ask": "#eb6834", "read only": "#1baf7a", "nothing": MUTED}
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for i, b in enumerate(reversed(BATCH)):
        c = collections.Counter(outcome(s["label"]["tool_calls"]) for s in by[b])
        left = 0
        for k in kinds:
            w = 100 * c[k] / len(by[b])
            ax.barh(i, w, left=left, color=kcol[k], edgecolor="white", linewidth=2, height=0.6, label=k if i == 0 else None)
            if w >= 4:
                ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            left += w
    ax.set_yticks(range(3), [f"batch {b}" for b in reversed(BATCH)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of gold answers (%)")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(False)
    save(fig, "04_answer_types.png", "What the correct answer is, by batch (train)")

    # 5. hearing vs reasoning (VAL: batch_table "final-turn words wrong")
    q = {"A": [92.6, 82.6, 32.3], "B": [88.1, 79.7, 43.0], "C": [62.6, 60.4, 54.7]}
    qn = {"A": [27, 23, 9], "B": [41, 60, 34], "C": [37, 73, 82]}
    groups = ["word-perfect transcript", "1-2 words wrong", "3+ words wrong"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(3)
    w = 0.26
    for j, b in enumerate(BATCH):
        bars = ax.bar(x + (j - 1) * w, q[b], width=w - 0.02, color=COL[b], label=f"batch {b}")
        for bar, n in zip(bars, qn[b]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.0f}", ha="center", fontsize=9, color=INK)
            ax.text(bar.get_x() + bar.get_width() / 2, 2, f"n={n}", ha="center", fontsize=7, color="white")
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 105)
    ax.set_ylabel("validation score")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "05_hearing_vs_reasoning.png", "Batch C stays low even when the transcript is perfect",
         "Score of validation samples grouped by wrong words in the final-turn transcript. In C the language model is the limit.")

    # 6. where the points go (VAL: lost_points / hash_diff)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
    lost = [("wrong final state (91 samples)", 23.6), ("right state, some calls wrong (50)", 5.4),
            ("right state, no call matched (16)", 4.1), ("gold empty, we acted (1)", 0.3)]
    bars = ax1.barh([n for n, _ in lost][::-1], [v for _, v in lost][::-1], color=MUTED, height=0.6)
    bars[-1].set_color("#d03b3b")
    label_bars(ax1, bars, "{:.1f}", horizontal=True, pad=0.3)
    ax1.set_xlabel("validation points lost (of 33.3)")
    ax1.set_title("Points lost, by kind", fontsize=10)
    ax1.set_xlim(0, 28)
    ax1.grid(axis="y", visible=False)
    brk = [("an item created or deleted wrongly", 45), ("balance", 10), ("progress minutes", 10), ("start / end time", 9),
           ("status", 8), ("time", 7), ("minutes", 6)]
    bars = ax2.barh([n for n, _ in brk][::-1], [v for _, v in brk][::-1], color="#2a78d6", height=0.6)
    label_bars(ax2, bars, "{:.0f}", horizontal=True, pad=0.5)
    ax2.set_xlabel("wrong-state samples where this field differs")
    ax2.set_title("What differs in the 91 wrong states", fontsize=10)
    ax2.set_xlim(0, 52)
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "06_where_points_go.png", "Most points are lost to a wrong final state, mostly numbers and created items",
         "Validation 66.71: 33.3 points lost. A sample can differ in several fields.")

    # 7. unseen apps (VAL: batch_table random / held halves)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
    pairs = [("B", 79.1, 69.3), ("C", 60.9, 56.0)]
    for i, (b, r, h) in enumerate(pairs):
        ax1.plot([h, r], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax1.scatter([r], [i], s=90, color=COL[b], zorder=2, edgecolor="white", linewidth=2)
        ax1.scatter([h], [i], s=90, color="white", zorder=2, edgecolor=COL[b], linewidth=2.5)
        ax1.text(r + 0.8, i, f"seen apps {r:.1f}", va="center", fontsize=9, color=INK)
        ax1.text(h - 0.8, i, f"unseen apps {h:.1f}", va="center", ha="right", fontsize=9, color=INK)
        ax1.text((r + h) / 2, i + 0.22, f"−{r - h:.1f}", ha="center", fontsize=9, color=INK2)
    ax1.set_yticks([0, 1], ["batch B", "batch C"])
    ax1.set_ylim(-0.6, 1.6)
    ax1.set_xlim(45, 90)
    ax1.set_xlabel("validation score (random half vs held-out-app half)")
    ax1.set_title("Inside one batch, unseen apps cost 5-10 points", fontsize=10)
    ax1.grid(axis="y", visible=False)
    share = [100 * np.mean([bool(set(apps(s)) - trapps) for s in pub if public_batch(s["id"]) == b]) for b in BATCH]
    bars = ax2.bar(BATCH, share, color=[COL[b] for b in BATCH], width=0.6)
    label_bars(ax2, bars, "{:.0f}%", pad=2)
    ax2.set_ylim(0, 110)
    ax2.set_title("Public samples with an app train never had", fontsize=10)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "07_unseen_apps.png", "New apps cost points, and public batches B and C are full of them")

    # 8. answer-key habits that flip between batches (gold, train)
    def share(pred_call, pred_ok, batches=BATCH):
        res = {}
        for b in batches:
            calls = [c for s in by[b] for c in s["label"]["tool_calls"] if pred_call(c, s)]
            res[b] = (sum(pred_ok(c) for c in calls), len(calls))
        return res
    habits = [
        ("calendar event: explicit 1-hour end", share(lambda c, s: c["name"] == "calendar.create_event" and (
            "end" not in c["arguments"] or c["arguments"]["end"][11:13] != c["arguments"]["start"][11:13] and
            int(c["arguments"]["end"][11:13]) - int(c["arguments"]["start"][11:13]) == 1 and
            c["arguments"]["end"][14:16] == c["arguments"]["start"][14:16]), lambda c: "end" in c["arguments"])),
        ("home rule has a title", share(lambda c, s: c["name"] == "home_automation.create_rule", lambda c: "title" in c["arguments"])),
        ("note has a title", share(lambda c, s: c["name"] == "notes.create", lambda c: "title" in c["arguments"])),
        ("phone call by contact id", share(lambda c, s: c["name"] == "phone.call", lambda c: "contact_id" in c["arguments"])),
        ("calendar listing with dates", share(lambda c, s: c["name"] == "calendar.list_events", lambda c: bool(c["arguments"]))),
        ("place search with 'near'", share(lambda c, s: c["name"] == "maps.search_place", lambda c: "near" in c["arguments"])),
        ("appliance start with a delay", share(lambda c, s: c["name"] == "appliances.start_cycle", lambda c: "delay_minutes" in c["arguments"])),
        ("contact search with a name", share(lambda c, s: c["name"] == "contacts.search", lambda c: bool(c["arguments"].get("query")))),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(habits))[::-1]
    for j, b in enumerate(BATCH):
        for yi, (name, res) in zip(y, habits):
            k, n = res[b]
            if not n:
                continue
            v = 100 * k / n
            ax.barh(yi + (1 - j) * 0.26, v, height=0.24, color=COL[b])
            ax.text(v + 1, yi + (1 - j) * 0.26, f"{k}/{n}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [h[0] for h in habits])
    ax.set_xlim(0, 112)
    ax.set_xlabel("share of gold calls with the habit (%)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL[b], label=f"batch {b}") for b in BATCH], ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="y", visible=False)
    save(fig, "08_habits_by_batch.png", "The answer key changes some habits between batches",
         "Calendar row counts only events with no other spoken duration. Missing bar = the tool never appears in that batch.")

    # 9. speech errors by slice (VAL: asr_slices, five-copy model out of fold)
    sl = [("all turns", 7.18, INK2), ("Northern words", 6.2, "#2a78d6"), ("Southern words", 10.2, "#2a78d6"),
          ("Central words", 7.9, "#2a78d6"), ("clean audio (≥50 dB)", 5.9, "#1baf7a"), ("noisy audio (<20 dB)", 9.6, "#1baf7a"),
          ("clip with silent tail", 6.3, "#eda100"), ("clip cut at speech end", 7.8, "#eda100"),
          ("history turns", 6.9, "#e87ba4"), ("final turns", 7.5, "#e87ba4")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    bars = ax1.barh([s[0] for s in sl][::-1], [s[1] for s in sl][::-1], color=[s[2] for s in sl][::-1], height=0.6)
    label_bars(ax1, bars, "{:.1f}%", horizontal=True, pad=0.15)
    ax1.set_xlim(0, 12)
    ax1.set_xlabel("word error rate (%)")
    ax1.set_title("Speech model: where it mishears", fontsize=10)
    ax1.grid(axis="y", visible=False)
    ds = [("unmarked", 73.7), ("Northern", 67.9), ("Southern", 58.9), ("Central (n=25)", 41.7)]
    bars = ax2.bar([d[0] for d in ds], [d[1] for d in ds], color=["#b4b2a9", "#2a78d6", "#2a78d6", "#2a78d6"], width=0.6)
    label_bars(ax2, bars, "{:.1f}", pad=1)
    ax2.set_ylim(0, 85)
    ax2.set_title("Validation score by dialect words in the request", fontsize=10)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "09_speech_errors.png", "Regional speech and noise are the hard cases",
         "Dialect is guessed from marker words (e.g. South 'nghen, hông, tui', Central 'mô, ni, rứa', North 'nhé, nhỉ, đấy').")

    # 10. audio is the same in train and public
    feats = list(csv.DictReader(open(feats_csv, encoding="utf-8")))
    split = lambda r: "train" if r["path"].startswith("train/") else "public"  # noqa: E731
    panels = [("snr_db", "signal-to-noise (dB)", (0, 80)), ("dur", "clip length (s)", (0, 20)),
              ("loud_db", "loudness, 95th-percentile frame (dB)", (-30, -10)), ("trail", "silence after speech (s)", (0, 1))]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    for ax, (k, t, lim) in zip(axes, panels):
        bins = np.linspace(lim[0], lim[1], 41)
        for name, c in (("train", "#2a78d6"), ("public", "#eb6834")):
            v = np.array([float(r[k]) for r in feats if split(r) == name])
            ax.hist(np.clip(v, lim[0], lim[1]), bins=bins, density=True, histtype="step", lw=2, color=c, label=name)
        ax.set_title(t, fontsize=10)
        ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "10_audio_same_both_splits.png", "The audio of train and public test is the same",
         f"All {len(feats):,} clips, curves scaled to equal area. No recording shift to adapt to.")

    # 11. batch C sub-styles along the ids (VAL: lost_points families)
    fam = [("F1 multi-intent\n(train 1-437)", 80.5, 62, "A"), ("F2 one request, wide apps\n(438-1094)", 66.4, 177, "B"),
           ("F3 follow-ups\n(1095-1532)", 59.2, 102, "C"), ("F4 mid history\n(1533-1642)", 77.1, 24, "C"),
           ("F5 long small talk\n(1643-1753)", 53.2, 21, "C")]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    bars = ax.bar([f[0] for f in fam], [f[1] for f in fam], color=[COL[f[3]] if f[3] != "B" else "#7f7d76" for f in fam], width=0.6)
    for bar, f in zip(bars, fam):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{f[1]:.1f}", ha="center", fontsize=9, color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, 3, f"n={f[2]}", ha="center", fontsize=8, color="white")
    ax.set_ylim(0, 95)
    ax.set_ylabel("validation score")
    ax.grid(axis="x", visible=False)
    save(fig, "11_styles_along_ids.png", "Follow-ups and long small-talk histories are the weakest styles",
         "Styles change along the id range. F2 mixes batch B and early batch C (grey). F5 ends in a wrong state 43% of the time.")

    # 12. ideas measured (VAL: post_rules / punct_rule / log.md)
    ideas = [("batch-habit rules (calendar end, rule title)", 0.35), ("retry empty contact search (Wave 1 re-check)", 0.08),
             ("rewrite contact search to stored name", 0.09), ("trailing punctuation like gold", 0.0),
             ("batch-A calendar listing without dates", 0.0)]
    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.barh([i[0] for i in ideas][::-1], [i[1] for i in ideas][::-1],
                   color=["#1baf7a" if v >= 0.3 else MUTED for _, v in ideas][::-1], height=0.55)
    label_bars(ax, bars, "{:+.2f}", horizontal=True, pad=0.01)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("change in validation score (points)")
    ax.grid(axis="y", visible=False)
    save(fig, "12_rules_tested.png", "Cheap rules are nearly used up",
         "Only the batch-habit rules move the score, and they cannot change the saved state.")


if __name__ == "__main__":
    main(sys.argv[1])
