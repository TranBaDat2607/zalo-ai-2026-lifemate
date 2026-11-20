"""Local copy of the official LifeMate score (the organisers' task statement, Evaluation).

"Identical" non-fuzzy arguments are compared strictly, so this can only be as harsh as the organisers' scorer or harsher.
"""
import unicodedata


def token_f1(a, b) -> float:
    ta = unicodedata.normalize("NFC", str(a).lower()).split()
    tb = unicodedata.normalize("NFC", str(b).lower()).split()
    if not ta or not tb:
        return float(ta == tb)
    left = list(tb)
    common = 0
    for t in ta:
        if t in left:
            left.remove(t)
            common += 1
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def _fuzzy_params(name: str, catalog: dict) -> set:
    props = catalog.get(name, {}).get("parameters", {}).get("properties", {})
    return {k for k, v in props.items() if v.get("x-match") == "fuzzy"}


def calls_match(sub: dict, gold: dict, catalog: dict) -> bool:
    if sub["name"] != gold["name"]:
        return False
    a, b = sub.get("arguments") or {}, gold.get("arguments") or {}
    if a.keys() != b.keys():
        return False
    fuzzy = _fuzzy_params(gold["name"], catalog)
    return all(token_f1(a[k], b[k]) > 0.6 if k in fuzzy else a[k] == b[k] for k in b)


def count_matches(sub_calls: list, gold_calls: list, catalog: dict) -> int:
    """Each call matches at most one call; order is ignored. Greedy, which is optimal unless fuzzy candidates overlap."""
    left = list(gold_calls)
    matched = 0
    for call in sub_calls:
        for i, gold in enumerate(left):
            if calls_match(call, gold, catalog):
                del left[i]
                matched += 1
                break
    return matched


def score_sample(sub_calls: list, sub_hash: str, gold_calls: list, gold_hash: str, catalog: dict) -> float:
    if not gold_calls:
        return float(not sub_calls)
    matched = count_matches(sub_calls, gold_calls, catalog)
    if sub_hash != gold_hash or matched == 0:
        return 0.0
    p, r = matched / len(sub_calls), matched / len(gold_calls)
    return 2 * p * r / (p + r)


def gold_hash(tool_env, split: str, item: dict, root) -> str:
    """Replays the gold calls in a separate session. For scoring train only, never during inference."""
    env = tool_env(split, item["id"], root=root)
    for call in item["label"]["tool_calls"]:
        env.call(call["name"], call["arguments"])
    return env.get_state_hash()
