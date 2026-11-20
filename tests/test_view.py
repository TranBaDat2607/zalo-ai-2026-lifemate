"""The readings view (lifemate.view): braces only where readings disagree in words, formatting never opens one, and
every option is a real phrase."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate import view  # noqa: E402
from lifemate.noise import ConfusionNoise  # noqa: E402

MAIN = "Thứ mai bảy giờ nhắc anh uống canxi nha."


def test_agreement_leaves_the_main_text_unchanged():
    assert view.build(MAIN, [MAIN] * 5) == MAIN
    # case, punctuation, digits against words, and particles are not disagreements
    same = ["thứ mai 7 giờ nhắc anh uống canxi", "Thứ mai bảy giờ, nhắc anh uống canxi nhé", "", None]
    assert view.build(MAIN, same) == MAIN


def test_substitutions_show_main_first_then_by_count():
    others = ["tối mai 7 giờ nhắc anh uống canxi", "Thôi mai bảy giờ nhắc anh uống canxi",
              "tối mai bảy giờ nhắc anh uống canxi", MAIN, MAIN]
    assert view.build(MAIN, others) == "{Thứ|tối|Thôi} mai bảy giờ nhắc anh uống canxi nha."


def test_dropped_and_added_words_take_a_neighbour():
    assert view.build("tôi không muốn đi", ["tôi muốn đi"]) == "{tôi không|tôi} muốn đi"
    assert view.build("tôi muốn đi", ["tôi không muốn đi"]) == "{tôi|tôi không} muốn đi"
    assert view.build("không muốn đi", ["muốn đi"]) == "{không muốn|muốn} đi"  # nothing before: the next word


def test_numbers_keep_their_written_form_and_trailing_punctuation_moves_out():
    got = view.build("Chuyển 128 triệu cho Mai.", ["chuyển một trăm hai tám triệu cho mai"])
    assert got == "Chuyển {128|một trăm hai tám} triệu cho Mai."
    got = view.build("bên nhà cung cấp.", ["bên nhà cuốn cất."])
    assert got == "bên nhà {cung cấp|cuốn cất}."


def test_caps_keep_the_most_disputed_spans():
    main = "một hai ba bốn năm sáu bảy tám"
    weak = ["một hai ba bốn năm sáu bảy chín"] * 2   # 'tám' disputed by two readings
    once = ["mốt hai ba bốn năm sáu bảy tám"]        # 'một' by one
    got = view.build(main, weak + once + [main, main], max_spans=1)
    assert got == "một hai ba bốn năm sáu bảy {tám|chín}"
    # a reading of particles only, or a different text altogether, says nothing and hides no other span
    odd = ["ờ ờ ờ ờ", "xin chào các bạn hôm nay trời đẹp quá", "một hai ba bốn năm sáu bảy chín"]
    assert view.build(main, odd) == "một hai ba bốn năm sáu bảy {tám|chín}"


def test_outcomes_keep_dropped_and_added_words():
    assert view._outcomes(["a", "b", "c"], ["a", "x", "y", "c"]) == ["a", "x y", "c"]
    assert view._outcomes(["a", "b", "c"], ["a", "c"]) == ["a", "", "c"]
    assert view._outcomes(["a", "b"], ["a", "b", "z"]) == ["a", "b z"]


def test_bank_replays_how_real_readings_heard_each_word():
    disputed = ("tối mai bảy giờ", ["thứ mai bảy giờ", "tối mai bảy giờ", "thôi mai bảy giờ", "tối mai bảy giờ"])
    rows = [disputed] * 3 + [("tối nay về", ["tối nay về"] * 4)]
    bank = view.ReadingsBank.from_rows(rows)
    assert bank.seen["tối"] == 4 and len(bank.bank["tối"]) == 3 and "mai" not in bank.bank
    rng = random.Random(0)
    views = [bank.view("Tối mai bảy giờ nhắc anh", rng) for _ in range(400)]
    braced = [v for v in views if "{" in v]
    assert 0.6 < len(braced) / 400 < 0.9  # 'tối' was disputed in 3 of its 4 occurrences
    assert all(v == "{Thứ|Tối|Thôi} mai bảy giờ nhắc anh" for v in braced)
    assert all(v == "Tối mai bảy giờ nhắc anh" for v in views if "{" not in v)  # unseen words: rare words were right


def test_unseen_words_take_a_rare_words_pattern_with_blurred_spellings():
    rows = [("gọi Khoa", ["gọi Khoa", "gọi Hoa", "gọi Toa"])]  # 'khoa' seen once: a rare word, disputed
    bank = view.ReadingsBank.from_rows(rows)
    main, others = bank.readings("gọi Quyên", random.Random(1))
    assert main == "gọi Quyên" and len(others) == 2
    assert all(o.startswith("gọi ") and o != "gọi Quyên" for o in others) and others[0] != others[1]


def test_view_spec_puts_braces_on_latest_requests_only(tmp_path):
    import json
    from lifemate.infer import load_transcripts
    main = {"a.wav": "Thứ mai bảy giờ", "h.wav": "Thứ mai bảy giờ"}
    other = {"a.wav": "tối mai bảy giờ", "h.wav": "tối mai bảy giờ"}
    for name, d in (("m.json", main), ("o.json", other)):
        (tmp_path / name).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    spec = f"view:{tmp_path / 'm.json'},{tmp_path / 'o.json'}"
    assert load_transcripts(spec, {"a.wav"}) == {"a.wav": "{Thứ|tối} mai bảy giờ", "h.wav": "Thứ mai bảy giờ"}
    assert load_transcripts(spec)["h.wav"] == "{Thứ|tối} mai bảy giờ"  # without the latest set: every clip
