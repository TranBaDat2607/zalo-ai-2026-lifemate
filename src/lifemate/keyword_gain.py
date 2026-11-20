"""Predicted score change of a transcript set, from the words the score depends on, on the 386 validation requests.

    python -m lifemate.keyword_gain outputs/exp7/asr_medium/medium_train.json outputs/a0-ctx/a0:reads+ctx

Each latest request falls in one group: transcript exactly right, only other words wrong, or a key word wrong (a word
of a gold argument value, a number or time word, or a digit). The 5 Nov 2026 regression on `soup_eval` (log.md:
"Score with perfect transcripts") puts a key word wrong at -26.1 points per sample and other words only at -19.2,
against an exactly right transcript. The predicted change is the mean over the requests of the change in those
effects; its interval comes from resampling the requests. The end-to-end test on the same 386 samples resolves about
+-2 points, so a predicted change below about +1 point cannot be confirmed there. Specs as in ``asr_compare``.
"""
import argparse
import json
import random
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from . import data
from .asr_compare import load_set

KEY, OTHER = -26.1, -19.2
NUMTIME = set(("không một mốt hai ba bốn tư năm lăm sáu bảy tám chín mười mươi trăm nghìn ngàn triệu tỷ rưỡi giờ phút "
               "giây sáng trưa chiều tối đêm mai nay qua kia mốt thứ tuần tháng ngày hôm chủ nhật").split())
EFFECT = {"right": 0.0, "other": OTHER, "key": KEY}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    for a, b in (("hoá", "hóa"), ("huỷ", "hủy"), ("khoá", "khóa"), ("thuỷ", "thủy")):
        s = s.replace(a, b)
    return s


def toks(s: str) -> list:
    return re.findall(r"\w+", norm(s))


def arg_words(calls: list) -> set:
    out = set()
    for c in calls:
        for v in c["arguments"].values():
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, str) and not re.match(r"^[a-z]+_\w+$", x):  # ids such as lock_992 are not spoken
                    out |= set(toks(x))
    return out


def group(item: dict, text: str) -> str:
    ref, hyp = toks(item["audio_text"]), toks(text)
    wrong = []
    for op, i1, i2, _, _ in SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            wrong += ref[i1:i2]
        elif op == "insert":
            wrong.append("<ins>")
    if not wrong:
        return "right"
    keys = arg_words(item["label"]["tool_calls"]) | NUMTIME | {w for w in ref if w.isdigit()}
    return "key" if any(w in keys for w in wrong) else "other"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="baseline transcripts: a JSON file or folder:pass")
    ap.add_argument("b", help="new transcripts: a JSON file or folder:pass[+fallback]")
    ap.add_argument("--rounds", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    a, b = load_set(args.a), load_set(args.b)
    splits = json.loads((Path(__file__).parent / "splits.json").read_text(encoding="utf-8"))
    ids = set(splits["random"]) | set(splits["apps"])
    items = [i for i in data.load_split("train", data.find_root("train")) if i["id"] in ids]
    items = [i for i in items if i["audio"] in a and i["audio"] in b]
    ga = [group(i, a[i["audio"]]) for i in items]
    gb = [group(i, b[i["audio"]]) for i in items]
    diffs = [EFFECT[y] - EFFECT[x] for x, y in zip(ga, gb)]
    rng = random.Random(args.seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(args.rounds))
    count = lambda g: {k: g.count(k) for k in ("right", "other", "key")}  # noqa: E731
    moves = {}
    for x, y in zip(ga, gb):
        if x != y:
            moves[f"{x}->{y}"] = moves.get(f"{x}->{y}", 0) + 1
    out = {"requests": len(items), "a": count(ga), "b": count(gb), "moves": moves,
           "predicted_change": round(sum(diffs) / len(diffs), 2),
           "ci95": [round(means[int(0.025 * args.rounds)], 2), round(means[int(0.975 * args.rounds) - 1], 2)]}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return out


if __name__ == "__main__":
    main()
