"""Context-aware decoding for the speech model: phrases the agent can already see, picked by how they sound.

A user turn is decoded twice. The first pass has no prompt. Then phrases from what the agent can see before that turn
(the user profile, earlier assistant replies, earlier tool calls and results and, for the latest request, the results of
the agent's own read calls) are compared with the first transcript without case and tone marks, and the closest ones
go into Whisper's previous-text prompt (``<|startofprev|>``) for the second pass.

Nothing here reads ``<split>_environment/``: every phrase comes from the sample record or from tool results the agent
received itself. Training uses the same functions, with the gold path's read results standing in for the agent's.
"""
import difflib
import json
import re
import unicodedata

# ids ("alm_01", "ord-2231"), urls, e-mail addresses, ISO dates and times: never spoken as written
_SKIP = re.compile(r"^[a-z0-9]+([_-][a-z0-9]+)+$|^https?:|@|^\d{4}-\d{2}-\d{2}|^\d{1,2}:\d{2}", re.I)
_PARTS = re.compile(r"[,;()\[\]/|\n]+")
_LEAD_NUMBER = re.compile(r"^[\d\s.,:+-]+")
_WORD = re.compile(r"[^\W_]+", re.U)
_VI_SYLLABLE = re.compile(r"^(b|c|ch|d|g|gh|gi|h|k|kh|l|m|n|ng|ngh|nh|p|ph|q|qu|r|s|t|th|tr|v|x)?[aeiouy]+(c|ch|m|n|ng|nh|p|t)?$")
SOURCE_RANK = {"read": 0, "history": 1, "profile": 2}
MAX_WORDS = 6  # longer strings are messages and descriptions, not names


def detone(text: str) -> str:
    """Lower case without tone marks or the bar of đ: what a misheard syllable usually keeps."""
    text = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def plain_words(text: str) -> list:
    return _WORD.findall(detone(text))


def _is_foreign(word: str) -> bool:
    return word.isascii() and (word.isupper() and len(word) >= 2 or not _VI_SYLLABLE.match(word.lower()))


def entity_spans(text: str) -> list:
    """Runs of capitalised words (not the first word of a sentence) and of non-Vietnamese words in free text."""
    spans, run, first = [], [], True
    for raw in text.split():
        word = raw.strip(".,?!:;…\"'()")
        keep = bool(word) and not any(c.isdigit() for c in word) and len(word) >= 2 and \
            (_is_foreign(word) or (word[0].isupper() and not first))
        if keep:
            run.append(word)
        if not keep or raw[-1] in ".,?!:;…":
            if run:
                spans.append(" ".join(run))
            run = []
        first = raw[-1] in ".?!…"
    if run:
        spans.append(" ".join(run))
    return spans


def _add(out: dict, phrase: str, source: str) -> None:
    phrase = _LEAD_NUMBER.sub("", phrase.strip(" .,:;!?\"'")).strip(" .,:;!?\"'")
    if not phrase or not any(c.isalpha() for c in phrase) or len(phrase) < 2:
        return
    key = phrase.lower()
    if key not in out or SOURCE_RANK[source] < SOURCE_RANK[out[key][1]]:
        out[key] = (phrase, source)


def add_value(obj, out: dict, source: str) -> None:
    """Every short string inside a JSON value (tool arguments, tool results, the profile)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "id" or key.endswith("_id") or key.endswith("_ids"):
                continue
            add_value(value, out, source)
    elif isinstance(obj, list):
        for value in obj:
            add_value(value, out, source)
    elif isinstance(obj, str) and not _SKIP.search(obj.strip()):
        for part in _PARTS.split(obj):
            part = part.strip()
            if not part:
                continue
            if len(part.split()) <= MAX_WORDS:
                _add(out, part, source)
            else:
                for span in entity_spans(part):
                    _add(out, span, source)


def turn_pools(item: dict) -> list:
    """For each user clip of a sample, in the order of ``asr_train.clips_of``: the phrases visible before it."""
    seen = {}
    add_value(item.get("context", {}).get("user_profile", {}), seen, "profile")
    pools = []
    for turn in item["history"]:
        if turn["role"] == "user":
            if turn.get("audio"):
                pools.append(dict(seen))
            continue
        for span in entity_spans(turn.get("text") or ""):
            _add(seen, span, "history")
        for call in turn.get("tool_calls") or []:
            add_value(call.get("arguments", {}), seen, "history")
        add_value(turn.get("tool_results") or [], seen, "history")
    pools.append(dict(seen))
    return pools


def read_results(calls: list, results: list, catalog: dict) -> dict:
    """Phrases from the results of read calls only: a write's result echoes its own arguments."""
    out = {}
    for call, result in zip(calls, results):
        if catalog.get(call.get("name"), {}).get("kind") == "read":
            add_value(result, out, "read")
    return out


def merge(*pools: dict) -> dict:
    out = {}
    for pool in pools:
        for key, (phrase, source) in pool.items():
            if key not in out or SOURCE_RANK[source] < SOURCE_RANK[out[key][1]]:
                out[key] = (phrase, source)
    return out


def similarity(words: list, phrase_words: list) -> float:
    """Best match of the phrase against any run of the transcript about as long, without case and tone marks."""
    k = len(phrase_words)
    target = " ".join(phrase_words)
    best = 0.0
    for n in {max(1, k - 1), k, k + 1}:
        for i in range(0, len(words) - n + 1):
            matcher = difflib.SequenceMatcher(None, " ".join(words[i:i + n]), target, autojunk=False)
            if matcher.real_quick_ratio() > best and matcher.quick_ratio() > best:
                best = max(best, matcher.ratio())
    return best


def select(pool: dict, first_pass: str, threshold: float = 0.7, single_threshold: float = 0.8, limit: int = 12) -> list:
    """Phrases of the pool that sound like a part of the first transcript, closest first."""
    words = plain_words(first_pass)
    scored = []
    for key, (phrase, source) in pool.items():
        pw = plain_words(phrase)
        if not pw or sum(len(w) for w in pw) < 3 or len(pw) > MAX_WORDS:
            continue
        score = similarity(words, pw)
        if score >= (single_threshold if len(pw) == 1 else threshold):
            scored.append((-score, SOURCE_RANK[source], phrase))
    return [phrase for _, _, phrase in sorted(scored)[:limit]]


def prompt_text(phrases: list) -> str:
    return ", ".join(phrases)


def fit_tokens(phrases: list, tokenizer, budget: int) -> list:
    """Drop the last phrases until the prompt takes at most ``budget`` tokens."""
    phrases = list(phrases)
    while phrases and len(tokenizer.encode(" " + prompt_text(phrases), add_special_tokens=False)) > budget:
        phrases.pop()
    return phrases


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
