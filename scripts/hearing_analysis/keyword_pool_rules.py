"""The word rules of keyword_pool.py (same definitions), importable without running that analysis."""
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from build_pairs import FILLERS  # noqa: E402
from lifemate.asr_eval import normal_tokens  # noqa: E402

TIME = set("sáng trưa chiều tối đêm khuya mai nay mốt qua hôm giờ phút rưỡi kém thứ tuần tháng năm ngày sau trước "
           "tới hai ba tư bốn năm sáu bảy tám chín mười mươi chủ nhật".split())
NUMW = set("không một mốt hai ba bốn tư năm lăm sáu bảy bẩy tám chín mười mươi trăm nghìn ngàn triệu tỷ linh lẻ rưỡi".split())


def toks(text):
    return [w for w in normal_tokens(text) if w not in FILLERS]


def value_tokens(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return set(normal_tokens(str(int(v)) if float(v).is_integer() else str(v)))
    if isinstance(v, str):
        return set(normal_tokens(v))
    if isinstance(v, list):
        return set().union(*[value_tokens(x) for x in v]) if v else set()
    if isinstance(v, dict):
        return set().union(*[value_tokens(x) for x in v.values()]) if v else set()
    return set()


def spans(gold, heard):
    sm = difflib.SequenceMatcher(a=gold, b=heard, autojunk=False)
    return [(gold[i1:i2], heard[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
