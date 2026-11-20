import pytest

from lifemate import asr_eval


@pytest.mark.parametrize("text,spoken", [
    ("5", "năm"),
    ("15", "mười lăm"),
    ("21", "hai mươi mốt"),
    ("24", "hai mươi tư"),
    ("50", "năm mươi"),
    ("95", "chín mươi lăm"),
    ("105", "một trăm linh năm"),
    ("500", "năm trăm"),
    ("1005", "một nghìn không trăm linh năm"),
    ("0912", "không chín một hai"),
])
def test_spell_number(text, spoken):
    assert " ".join(asr_eval.spell_number(text)) == spoken


def test_normalised_wer_ignores_format_but_strict_does_not():
    ref, hyp = "Nạp 50 ngàn cho điều hoà.", "nạp năm mươi nghìn cho điều hòa"
    row = asr_eval.score_turn(ref, hyp)
    assert row["norm_errors"] == 0 and row["exact"]
    assert row["strict_errors"] > 0


def test_edit_distance():
    assert asr_eval.edit_distance(list("abc"), list("axcd")) == 2
    assert asr_eval.edit_distance([], list("ab")) == 2


def test_critical_terms_skip_sentence_start():
    terms = asr_eval.critical_terms("Nhắn Sếp Dũng qua SMS lúc 9 giờ. Gửi file cho Lan.")
    assert terms["number"] == ["9"]
    assert terms["name"] == ["Sếp", "Dũng", "Lan"]
    assert terms["foreign"] == ["SMS", "file"]


def test_term_hits_accept_short_number_reading_and_count_misses():
    assert asr_eval.term_hits("Đổ RON 95 nhé", "đổ ron chín lăm nhé")["number"] == (1, 1)
    hits = asr_eval.term_hits("Chuyển 500 cho Lan", "chuyển năm trăm cho lâm")
    assert hits["number"] == (1, 1) and hits["name"] == (0, 1)


def test_turns_include_history_users_only():
    item = {"id": "a", "audio": "x.wav", "audio_text": "c",
            "history": [{"role": "user", "audio": "u.wav", "audio_text": "b"}, {"role": "assistant", "text": "ok"}]}
    assert [t["audio"] for t in asr_eval.turns_of([item])] == ["u.wav", "x.wav"]
