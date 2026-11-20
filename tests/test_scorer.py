import pytest

from lifemate import data, prompts, scorer

CATALOG = {"sms.send": {"parameters": {"properties": {"contact_id": {}, "content": {"x-match": "fuzzy"}}}}}


def call(content="xin chao ban"):
    return {"name": "sms.send", "arguments": {"contact_id": "c1", "content": content}}


def test_empty_gold_needs_empty_submission():
    assert scorer.score_sample([], "x", [], "y", CATALOG) == 1
    assert scorer.score_sample([call()], "x", [], "x", CATALOG) == 0


def test_wrong_hash_scores_zero():
    assert scorer.score_sample([call()], "x", [call()], "y", CATALOG) == 0


def test_extra_read_matches_focus_example():
    gold = [call(), {"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]
    assert scorer.score_sample(gold + [{"name": "c", "arguments": {}}], "h", gold, "h", CATALOG) == pytest.approx(6 / 7)


def test_fuzzy_threshold_is_strict():
    assert scorer.calls_match(call("xin chao ban"), call("xin chao ban nhe"), CATALOG)  # F1 0.857
    assert not scorer.calls_match(call("xin chao"), call("tam biet ban"), CATALOG)


def test_gold_replay_scores_100_on_train():
    try:
        root = data.find_root("train")
    except FileNotFoundError:
        pytest.skip("data package not available")
    items = data.load_split("train", root)
    catalog = data.load_catalog("train", root)
    tool_env = data.import_tool_env("train", root)
    for item in items:
        h = scorer.gold_hash(tool_env, "train", item, root)
        assert scorer.score_sample(item["label"]["tool_calls"], h, item["label"]["tool_calls"], h, catalog) == 1, item["id"]


def test_prompt_lists_installed_tools_and_the_latest_request():
    root = data.find_root("train")
    catalog = data.load_catalog("train", root)
    for item in data.load_split("train", root)[:200]:
        text = prompts.prompt_text(item, prompts.gold_transcripts(item), catalog)
        assert prompts.tool_names(item, catalog)[-1] == "sys.ask_missing_info"
        assert "sys.ask_missing_info(" in text
        assert text.endswith(prompts.turn("user", item["audio_text"]))
