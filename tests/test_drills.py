"""Time/number drills (lifemate.drills, improve_plan.md step 4): the generator's values, its readers on hand-checked
phrases, the train-gold conventions, and the training encoding (loss on the answers only)."""
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate import drills, prompts  # noqa: E402

NOW = ["2026-10-01T09:00:00+07:00", "2026-11-14T19:30:00+07:00", "2026-12-08T16:40:00+07:00", "2026-09-20T09:00:00+07:00"]


def test_readers_on_hand_checked_phrases():
    assert drills.parse_number("hai triệu tư") == 2_400_000
    assert drills.parse_number("hai triệu rưỡi") == 2_500_000
    assert drills.parse_number("1,5 triệu") == 1_500_000
    assert drills.parse_number("năm chục nghìn") == 50_000
    assert drills.parse_number("một trăm rưỡi") == 150
    assert drills.parse_number("ba lăm") == 35
    assert drills.parse_number("hai mốt") == 21
    assert drills.parse_number("sáu trăm bốn chục nghìn") == 640_000
    assert drills.parse_clock("chín giờ kém mười lăm", "sáng") == (8, 45)
    assert drills.parse_clock("bảy rưỡi", "tối") == (19, 30)
    assert drills.parse_clock("mười giờ", "tối") == (22, 0)
    assert drills.parse_clock("mười một giờ", "khuya") == (23, 0)
    assert drills.parse_clock("một giờ", "trưa") == (13, 0)
    assert drills.parse_clock("9 giờ 15", "sáng") == (9, 15)
    assert drills.parse_duration("một tiếng rưỡi") == 90
    assert drills.parse_duration("nửa tiếng") == 30
    assert drills.parse_duration("hai tiếng mười lăm phút") == 135


def test_every_drill_reads_back_and_is_deterministic():
    a = drills.generate(NOW, 600, seed=5)
    b = drills.generate(NOW, 600, seed=5)
    assert a == b
    ds = [d for p in a for d in p["drills"]]
    assert len(ds) == 600
    assert not [d for d in ds if drills.verify(d)]
    for d in ds:
        assert d["target"] and all(isinstance(k, str) for k in d["target"])
        json.dumps(d, ensure_ascii=False)


def test_weekday_dates_are_the_next_such_day():
    rng = random.Random(3)
    for _ in range(300):
        now = datetime.fromisoformat(rng.choice(NOW))
        d = drills.drill_date(drills.Ctx(now, rng))
        if d["form"] != "weekday":
            continue
        day = datetime.fromisoformat(d["target"]["date"]).date()
        assert 1 <= (day - now.date()).days <= 6
        word = next(w for w in drills.WEEKDAY_WORDS if w in d["text"].lower())
        assert drills.WEEKDAY_WORDS.index(word) == day.weekday()


def test_clock_drills_are_in_the_future_and_match_their_words():
    rng = random.Random(4)
    for _ in range(300):
        now = datetime.fromisoformat(rng.choice(NOW))
        d = drills.drill_clock(drills.Ctx(now, rng))
        (param, value), = d["target"].items()
        t = datetime.fromisoformat(value)
        assert t > now
        _, phrase, part, h, m = d["checks"][0]
        assert (t.hour, t.minute) == (h, m) == drills.parse_clock(phrase, part)


def test_rules_keep_hh_mm():
    rng = random.Random(6)
    for _ in range(100):
        d = drills.drill_rule(drills.Ctx(datetime.fromisoformat(NOW[0]), rng))
        value = d["target"]["time"]
        assert len(value) == 5 and value[2] == ":"
        _, phrase, part, h, m = d["checks"][0]
        assert value == f"{h:02d}:{m:02d}"


def test_avoided_ngrams_never_appear():
    packs = drills.generate(NOW, 200, seed=8)
    seen = set().union(*(drills.ngrams(d["text"]) for p in packs for d in p["drills"]))
    avoid = set(list(seen)[:50])
    again = drills.generate(NOW, 200, seed=8, avoid=avoid)
    assert not any(drills.ngrams(d["text"]) & avoid for p in again for d in p["drills"])


def _tok(text):
    return [ord(c) for c in text]


def test_pack_encoding_puts_the_loss_on_the_answers_only():
    pack = drills.generate(NOW, 5, seed=2, per_pack=(5, 5))[0]
    ids, labels = prompts.encode_drill_pack(pack, _tok)
    assert len(ids) == len(labels)
    learned = "".join(chr(t) for t in labels if t != -100)
    want = "".join(drills.answer_text(d["target"]) + prompts.END for d in pack["drills"])
    assert learned == want
    text = "".join(chr(t) for t in ids)
    assert text.count(prompts.GEN_PREFIX) == len(pack["drills"])
    assert f"- Bây giờ: {pack['now']}" in text and "- Các ngày tới: " in text
    for d in pack["drills"]:
        assert d["text"] in text
        assert drills.DRILL_TAG.format(params=", ".join(d["target"])) in text
        if d.get("data"):
            assert prompts.DATA_PREFIX + prompts.dumps(d["data"]) in text


def test_bundled_fold0_file_passes_verify():
    path = Path(prompts.__file__).parent / "drills_fold0.jsonl"
    packs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ds = [d for p in packs for d in p["drills"]]
    assert len(packs) == 600 and len(ds) == 3000
    assert not [d for d in ds if drills.verify(d)]
    gate = json.loads((path.parent / "drills_fold0_gate.json").read_text(encoding="utf-8"))
    assert gate["decision"] == "PASS"
    assert not any(p in d["target"] for d in ds for p in ("steps", "valid_until"))  # params of held-out apps only
