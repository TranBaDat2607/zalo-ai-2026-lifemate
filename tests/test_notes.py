"""Worked notes: where the note sits in the training text, and the first turn's token limit at inference."""
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _item():
    return {"id": "train_9999", "audio": "a.wav", "audio_text": "Bật đèn phòng khách", "history": [],
            "context": {"current_time": "2026-11-07T09:00:00+07:00", "user_profile": {}, "installed_apps": {
                "smart_home": {"tools": ["smart_home.get_status"]}}},
            "label": {"tool_calls": [{"name": "smart_home.get_status", "arguments": {}}],
                      "tool_results": [{"ok": True, "devices": []}]}}


def _catalog():
    return {"smart_home.get_status": {"name": "smart_home.get_status", "description": "Trạng thái", "kind": "read",
                                      "parameters": {"properties": {}}},
            "sys.ask_missing_info": {"name": "sys.ask_missing_info", "description": "Hỏi", "kind": "meta",
                                     "parameters": {"properties": {}}}}


def _decode(ids):
    return "".join(chr(i) for i in ids)


def test_note_follows_the_restated_request_and_is_trained():
    from lifemate import prompts
    tok = lambda t: [ord(c) for c in t]  # noqa: E731
    item, catalog = _item(), _catalog()
    texts = {"a.wav": "bật đèn phòng khác"}
    ids0, labels0 = prompts.encode_conversation(item, texts, catalog, tok, denoise=True)
    ids1, labels1 = prompts.encode_conversation(item, texts, catalog, tok, denoise=True, note="Cần device_id: smart_home.get_status.")
    text1 = _decode(ids1)
    assert "<think>\nBật đèn phòng khách\nCần device_id: smart_home.get_status.\n</think>\n\n<tool_call>" in text1
    trained = _decode([i for i, l in zip(ids1, labels1) if l != -100])
    assert trained.startswith("Bật đèn phòng khách\nCần device_id")
    # without a note nothing changes, and the note adds exactly its own text plus one newline
    assert len(ids1) - len(ids0) == len("\nCần device_id: smart_home.get_status.")
    assert _decode(ids0) == text1.replace("\nCần device_id: smart_home.get_status.", "")
    # without denoise the note is ignored (there is no think block to hold it)
    ids2, _ = prompts.encode_conversation(item, texts, catalog, tok, denoise=False, note="x")
    ids3, _ = prompts.encode_conversation(item, texts, catalog, tok, denoise=False)
    assert ids2 == ids3


def test_note_is_renamed_with_masked_tool_names():
    import random
    from lifemate import prompts
    tok = lambda t: [ord(c) for c in t]  # noqa: E731
    item, catalog = _item(), _catalog()
    rename = prompts.renamer(prompts.masked_names(item, catalog, random.Random(1)))
    ids, _ = prompts.encode_conversation(item, {"a.wav": "x"}, catalog, tok, denoise=True, rename=rename,
                                         note="Đọc smart_home.get_status trước.")
    assert "smart_home.get_status" not in _decode(ids)


class _LimitGen:
    """Records the token limit of each generate() call; the first turn carries a note before its call."""
    max_new_tokens = 256
    stop_id = -1

    def __init__(self, turns):
        self.turns, self.limits = list(turns), []
        self.tokenizer = self

    def encode(self, text):
        return [0] * len(text)

    def decode(self, ids, skip_special_tokens=False):
        return self._last

    def generate(self, ids, cache, fed, limit=0):
        self.limits.append(limit)
        self._last = self.turns.pop(0)
        return [1, self.stop_id], cache, fed


class _Env:
    def __init__(self):
        self.calls = []

    def validate(self, name, arguments):
        return {"ok": True, "error": None}

    def call(self, name, arguments):
        self.calls.append(name)
        return {"ok": True}


def _call(name, **arguments):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def test_first_turn_gets_its_own_token_limit():
    infer = importlib.import_module("lifemate.infer")
    first = "Bật đèn\nCần device_id: smart_home.get_status rồi smart_home.light_set.\n</think>\n\n" + _call("smart_home.get_status")
    gen = _LimitGen([first, _call("smart_home.light_set", device_id="l1"), "Xong."])
    env = _Env()
    stats = infer.run_agent(gen, env, "prompt", time.monotonic() + 60, 8, denoise=True, first_tokens=512)
    assert env.calls == ["smart_home.get_status", "smart_home.light_set"]
    assert gen.limits == [512, 0, 0] and stats["stop"] == "done"
    gen = _LimitGen([_call("smart_home.get_status"), "Xong."])  # without first_tokens every turn uses the default
    infer.run_agent(gen, _Env(), "prompt", time.monotonic() + 60, 8, denoise=True)
    assert gen.limits == [0, 0]


# ---------------------------------------------------------------- notes v2: kinds, branch-blind rules, package-v2
def _v2():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "notes"))
    return importlib.import_module("check_notes_v2")


def _ctx(**kw):
    ctx = {"primary": "K0", "kinds": ["K0"], "blind": False, "decision": None, "first_write": None,
           "request": "huỷ gói nghe nhạc giúp tôi", "user_text": "huỷ gói nghe nhạc giúp tôi",
           "allowed_params": {}, "gold_values": {}, "now_t": (13, 10), "now_d": (11, 8), "gold_times": set(),
           "gold_dates": set(), "said": set(),
           "params_of": {"subscriptions.list": set(), "subscriptions.cancel": {"subscription_id"},
                         "email.archive": {"email_id", "action"}, "reminders.create": {"time", "content"},
                         "ride_hailing.book": {"vehicle", "pickup"}},
           "enums_of": {"email.archive": {"action": {"archive", "delete"}}, "ride_hailing.book": {"vehicle": {"bike", "car4"}}}}
    ctx.update(kw)
    return ctx


def _k1(**kw):
    base = dict(primary="K1", kinds=["K1"], blind=True, decision=("subscriptions.cancel", "subscription_id"),
                allowed_params={"subscriptions.cancel": {"subscription_id"}})
    base.update(kw)
    return _ctx(**base)


GOOD_K1 = ("subscriptions.list tìm gói nghe nhạc; khớp đúng một thì subscriptions.cancel, không có hoặc nhiều thì "
           "sys.ask_missing_info cho subscriptions.cancel, params subscription_id.")


def test_v2_good_k1_note_passes():
    assert _v2().extra_checks(GOOD_K1, _k1()) == []


def test_v2_banned_phrases():
    v2 = _v2()
    bad = "subscriptions.list; không xác định được thì sys.ask_missing_info cho subscriptions.cancel, params subscription_id."
    assert any("banned phrase" in p for p in v2.extra_checks(bad, _k1()))
    assert any("banned phrase" in p for p in v2.extra_checks("subscriptions.list rồi chỉ đọc.", _ctx()))  # everywhere
    for phrase in ("tìm gói, vẫn còn hai gói nên", "đã có sẵn", "không có nên thôi"):
        note = GOOD_K1 + " " + phrase
        assert v2.extra_checks(note, _k1()), phrase
    # the same words are fine in a sample that is not branch-blind, except the ones banned everywhere
    assert v2.extra_checks("hoá đơn đã có trong lịch; subscriptions.list.", _ctx()) == []


def test_v2_k1_needs_both_branches_tool_param_and_ask():
    v2 = _v2()
    one = "subscriptions.list tìm gói rồi subscriptions.cancel, params subscription_id."
    probs = v2.extra_checks(one, _k1())
    assert any("one branch only" in p for p in probs) and any("sys.ask_missing_info" in p for p in probs)
    no_param = GOOD_K1.replace(", params subscription_id", "")
    assert any("parameter subscription_id" in p for p in v2.extra_checks(no_param, _k1()))
    no_tool = GOOD_K1.replace("subscriptions.cancel", "subscriptions.list")
    assert any("decision tool" in p for p in v2.extra_checks(no_tool, _k1()))


def test_v2_k2_both_branches_and_write_tool():
    v2 = _v2()
    ctx = _ctx(primary="K2", kinds=["K2"], blind=True, first_write="reminders.create")
    good = "weather.get_forecast xem có mưa không; mưa thì reminders.create nhắc, không mưa thì thôi."
    assert v2.extra_checks(good, ctx) == []
    assert any("one branch" in p for p in v2.extra_checks("weather.get_forecast, mưa thì reminders.create.", ctx))
    assert any("write tool" in p for p in v2.extra_checks("weather.get_forecast xem mưa không; mưa thì nhắc, không thì thôi.", ctx))
    assert any("branch-revealing" in p for p in v2.extra_checks(good + " Chắc là mưa.", ctx))


def test_v2_k3_selection_and_no_count():
    v2 = _v2()
    ctx = _ctx(primary="K3", kinds=["K3"], blind=True, first_write="reminders.create", user_text="tắt hết đèn tầng 2")
    assert any("selected" in p for p in v2.extra_checks("smart_home.get_status; reminders.create, không thì thôi.", ctx))
    counted = "smart_home.get_status; reminders.create tắt từng đèn, 3 đèn, không thì giữ."
    assert any("number 3" in p for p in v2.extra_checks(counted, ctx))
    ok = "smart_home.get_status tầng 2; reminders.create tắt từng đèn, trừ phòng có quạt thì giữ, không thì tắt."
    assert not [p for p in v2.extra_checks(ok, ctx) if "number" in p]


def test_v2_current_time_and_date_need_a_use():
    v2 = _v2()
    assert any("current time" in p for p in v2.extra_checks("Bây giờ 13:10. reminders.create.", _ctx()))
    assert not v2.extra_checks("13:10 + 45 phút = 13:55. reminders.create.", _ctx(gold_times={(13, 55)}))
    assert any("today's date" in p for p in v2.extra_checks("Hôm nay 8/11. subscriptions.list.", _ctx()))
    assert not v2.extra_checks("Hôm nay 8/11. subscriptions.list.", _ctx(gold_dates={(11, 8)}))


def test_v2_parameter_names_must_be_set_by_the_key():
    v2 = _v2()
    note = "reminders.create với time và content."
    ctx = _ctx(allowed_params={"reminders.create": {"content"}})
    probs = v2.extra_checks(note, ctx)
    assert any("parameter time" in p for p in probs) and not any("parameter content" in p for p in probs)
    # a branch-blind note may name only the decision parameter and the shown call's parameters
    probs = v2.extra_checks(GOOD_K1.replace("params subscription_id", "params subscription_id, vehicle") + " ride_hailing.book vehicle.", _k1())
    assert any("parameter vehicle" in p for p in probs)


def test_v2_option_values_must_be_the_keys():
    v2 = _v2()
    ctx = _ctx(gold_values={"email.archive": {"action": {"delete"}}}, allowed_params={"email.archive": {"action", "email_id"}})
    assert any("option value archive" in p for p in v2.extra_checks("email.archive action archive.", ctx))
    assert not v2.extra_checks("email.archive action delete.", ctx)
    blind = _k1()
    assert any("option value car4" in p for p in v2.extra_checks(GOOD_K1 + " ride_hailing.book car4.", blind))
    spoken = _k1(user_text="đặt xe car4", request="đặt xe car4")
    assert not [p for p in v2.extra_checks(GOOD_K1 + " ride_hailing.book car4.", spoken) if "option" in p]


def test_v2_kinds_from_gold_calls():
    from notes_v2_common import analyze
    cat = {"contacts.search": {"name": "contacts.search", "kind": "read", "parameters": {"properties": {"query": {}}}},
           "sms.send": {"name": "sms.send", "kind": "write", "parameters": {"properties": {
               "contact_id": {"x-ref": "contacts"}, "content": {}}}},
           "weather.get_forecast": {"name": "weather.get_forecast", "kind": "read", "parameters": {"properties": {"date": {}}}},
           "sys.ask_missing_info": {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"properties": {}}}}

    def item(request, calls, results):
        it = _item()
        it.update({"audio_text": request, "label": {"tool_calls": calls, "tool_results": results}})
        return it
    act = item("Nhắn Lan là anh tới trễ", [{"name": "contacts.search", "arguments": {"query": "Lan"}},
               {"name": "sms.send", "arguments": {"contact_id": "c_77", "content": "tới trễ"}}],
               [{"ok": True, "contacts": [{"id": "c_77"}]}, {"ok": True}])
    ask = item("Nhắn Lan là anh tới trễ", [{"name": "contacts.search", "arguments": {"query": "Lan"}},
               {"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}],
               [{"ok": True, "contacts": [{"id": "c_1"}, {"id": "c_2"}]}, {"ok": True}])
    a, b = analyze(act, cat), analyze(ask, cat)
    assert a["primary"] == b["primary"] == "K1" and a["decision"] == ("sms.send", "contact_id") == b["decision"]
    assert a["cut"] == b["cut"] == 1  # the writer sees only the first read: act and ask look the same
    cond = item("Mai mưa thì nhắn Lan là anh tới trễ", [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-08"}}],
                [{"ok": True, "rain": False}])
    assert analyze(cond, cat)["primary"] == "K2" and analyze(cond, cat)["cut"] == 1
    plain = item("Thời tiết mai thế nào", [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-08"}}],
                 [{"ok": True, "rain": False}])
    assert analyze(plain, cat)["primary"] == "K0"


def test_notes_v2_package_flag():
    from pathlib import Path as P
    train = importlib.import_module("lifemate.train")
    root = P(train.__file__).parent
    assert train.notes_path("package") == root / "notes.jsonl"          # v1 stays the default package
    assert train.notes_path("package-v2") == root / "notes_v2.jsonl"
    assert train.notes_path("/tmp/x.jsonl") == P("/tmp/x.jsonl")
