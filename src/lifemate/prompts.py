"""The prompt of one sample, written by hand as ChatML so that it is compact and token-prefix stable.

Inference and training share these functions so that the prompt cannot drift. A conversation is a list of segments
``(role, text, trainable)``; the agent loop appends to it, the trainer sets labels on the trainable ones.
"""
import json
import re
from datetime import datetime, timedelta

GEN_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
# With denoising, the first turn opens its think block and first restates what the user really said.
THINK_OPEN = "<|im_start|>assistant\n<think>\n"
THINK_CLOSE = "\n</think>\n\n"
END = "<|im_end|>\n"
DONE = "Xong."
_WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
_SHORT_DAYS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
_ID_SOURCE = re.compile(r"^ID [^,]*, lấy từ ")  # "ID thiết bị, lấy từ smart_home.get_status" -> "từ smart_home.get_status"

SYSTEM = """Bạn là trợ lý giọng nói tiếng Việt, xử lý yêu cầu mới nhất của người dùng bằng tool. Lời người dùng lấy từ nhận dạng giọng nói nên có thể sai chữ: hãy đoán đúng ý theo ngữ cảnh.
Mỗi lượt gọi đúng một tool:
<tool_call>
{"name":"app.tool","arguments":{...}}
</tool_call>
Không cần gọi thêm thì trả lời: Xong.
- Gọi đúng những gì cần: thừa hay thiếu đều mất điểm.
- Cần ID thì gọi tool đọc trước. Thông tin đã có trong lịch sử thì không đọc lại.
- Có điều kiện: đọc để kiểm tra rồi chỉ làm nhánh đúng.
- Thiếu tham số bắt buộc hoặc nhiều kết quả không phân biệt được: gọi sys.ask_missing_info với tool định gọi và các tham số còn thiếu xếp a-z, rồi dừng.
- Không app nào làm được, chỉ trò chuyện, hoặc người dùng rút lại: không gọi tool.
- Người dùng tự sửa lời ("à không", "à thôi"): theo ý sau cùng.
- Thời gian đủ giây và offset (2026-11-11T05:30:00+07:00); tiền và số lượng là số nguyên.
- Tên liên hệ viết như trong danh bạ; nội dung tin nhắn, ghi chú ngắn gọn, sát lời người dùng."""


def _short_type(name: str, spec: dict) -> str:
    """One parameter as compact text. Repeated boilerplate descriptions collapse into a type tag."""
    kind = spec.get("type")
    desc = spec.get("description", "") or ""
    tag = {"integer": "int", "number": "num", "boolean": "bool", "array": "list"}.get(kind, "")
    if "ISO-8601" in desc:
        tag = "datetime"
        desc = desc.split(";", 1)[1].strip() if ";" in desc else ""
    elif desc.startswith("YYYY-MM-DD"):
        tag, desc = "date", ""
    elif "Số nguyên" in desc:
        head = desc.split("Số nguyên", 1)[0].strip(" .")
        desc = head
    if spec.get("format") == "date-time":
        tag = "datetime"
    elif spec.get("format") == "date":
        tag = "date"
    elif spec.get("format") == "time":
        tag = "HH:MM"
    if spec.get("enum"):
        tag = "|".join(str(x) for x in spec["enum"])
    items = spec.get("items")
    if isinstance(items, dict) and items.get("type") == "object":
        fields = ",".join(k + ("*" if k in items.get("required", []) else "") for k in items.get("properties", {}))
        tag = f"list[{{{fields}}}]"
    elif isinstance(items, dict) and items.get("enum"):
        tag = "list[" + "|".join(items["enum"]) + "]"
    text = name
    if tag:
        text += f":{tag}"
    if desc:
        text += f" ({_ID_SOURCE.sub('từ ', desc)})"
    return text


def tool_line(tool: dict) -> str:
    params = tool.get("parameters") or {}
    required = set(params.get("required") or [])
    parts = []
    for name, spec in (params.get("properties") or {}).items():
        parts.append(_short_type(name + ("*" if name in required else ""), spec))
    line = f"{tool['name']}({', '.join(parts)}) - {tool.get('description', '')}"
    groups = params.get("x-one-of")
    if groups:
        line += " Bắt buộc một trong: " + " / ".join("+".join(g) for g in groups) + "."
    return line


def tool_names(item: dict, catalog: dict, drop_apps: frozenset = frozenset()) -> list:
    names = [t for a, app in item["context"]["installed_apps"].items() if a not in drop_apps for t in app["tools"]]
    names = [n for n in names if n in catalog and n != "sys.ask_missing_info"]
    return names + ["sys.ask_missing_info"]


def calendar_hint(now_iso: str) -> str:
    """Date arithmetic is the commonest slip of a small model, so the prompt spells out the next two weeks."""
    now = datetime.fromisoformat(now_iso)
    days = []
    for delta in range(0, 15):
        d = now + timedelta(days=delta)
        label = {0: " hôm nay", 1: " mai"}.get(delta, "")
        days.append(f"{_SHORT_DAYS[d.weekday()]} {d.day}/{d.month}{label}")
    return ", ".join(days)


def system_text(item: dict, catalog: dict, drop_apps: frozenset = frozenset()) -> str:
    ctx = item["context"]
    profile = ctx.get("user_profile", {})
    now = ctx["current_time"]
    lines = [SYSTEM, "", "Bối cảnh:", f"- Bây giờ: {now} ({_WEEKDAYS[datetime.fromisoformat(now).weekday()]})",
             f"- Các ngày tới: {calendar_hint(now)}",
             f"- Người dùng: {json.dumps(profile, ensure_ascii=False)}"]
    if ctx.get("location"):
        lines.append(f"- Vị trí: {json.dumps(ctx['location'], ensure_ascii=False)}")
    lines += ["", "Tool (* là bắt buộc):"] + [tool_line(catalog[n]) for n in tool_names(item, catalog, drop_apps)]
    return "\n".join(lines)


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def call_text(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def result_text(result: dict) -> str:
    return "<tool_response>\n" + dumps(result) + "\n</tool_response>"


def turn(role: str, text: str) -> str:
    return f"<|im_start|>{role}\n{text}{END}"


def history_segments(item: dict, transcripts: dict) -> list:
    """(text, call) pieces of the history. ``call`` is set on the text of a tool call the assistant made, which has the
    same form as a generated turn, so the trainer can also learn from it."""
    out = []
    for t in item["history"]:
        if t["role"] == "user":
            out.append((turn("user", transcripts[t["audio"]]), None))
            continue
        calls, results = t.get("tool_calls") or [], t.get("tool_results") or []
        for i, call in enumerate(calls):
            out.append((GEN_PREFIX, None))
            out.append((call_text(call["name"], call.get("arguments") or {}) + "<|im_end|>", call))
            out.append(("\n", None))
            if i < len(results):
                out.append((turn("user", result_text(results[i])), None))
        out.append((GEN_PREFIX + (t.get("text", "") or DONE) + END, None))
    return out


def history_text(item: dict, transcripts: dict) -> str:
    return "".join(text for text, _ in history_segments(item, transcripts))


def prompt_text(item: dict, transcripts: dict, catalog: dict, drop_apps: frozenset = frozenset()) -> str:
    """Everything up to and including the latest request; the agent loop generates from here."""
    return (turn("system", system_text(item, catalog, drop_apps)) + history_text(item, transcripts)
            + turn("user", transcripts[item["audio"]]))


def step_text(name: str, arguments: dict, result: dict) -> str:
    """A finished call and its result, without the generation prefix."""
    return f"{GEN_PREFIX}{call_text(name, arguments)}{END}" + turn("user", result_text(result))


def gold_transcripts(item: dict) -> dict:
    turns = {t["audio"]: t["audio_text"] for t in item["history"] if t["role"] == "user"}
    return {**turns, item["audio"]: item["audio_text"]}


def apps_to_drop(item: dict) -> list:
    """Installed apps the gold calls never touch, largest first: removing them cannot change the right answer."""
    used = {c["name"].split(".")[0] for c in item["label"]["tool_calls"]}
    used |= {c["arguments"].get("tool_name", ".").split(".")[0] for c in item["label"]["tool_calls"] if c["name"] == "sys.ask_missing_info"}
    apps = item["context"]["installed_apps"]
    return sorted((a for a in apps if a not in used), key=lambda a: (-len(apps[a]["tools"]), a))


def final_call_policy(items: list):
    """A predicate for history calls worth learning from.

    History calls often pass filters that final-turn gold avoids, for example smart_home.get_status with a room. A call
    only teaches the right habit if its set of arguments is also common for that tool in final-turn gold.
    """
    seen = {}
    for item in items:
        for call in item["label"]["tool_calls"]:
            keys = tuple(sorted(call["arguments"]))
            seen.setdefault(call["name"], {})
            seen[call["name"]][keys] = seen[call["name"]].get(keys, 0) + 1

    def ok(call: dict) -> bool:
        counts = seen.get(call["name"])
        if not counts or sum(counts.values()) < 5:
            return True
        return counts.get(tuple(sorted(call.get("arguments") or {})), 0) / sum(counts.values()) >= 0.1

    return ok


_SYLLABLES = [c + v for c in "bdghklmnprstvx" for v in "aeiou"]


def masked_names(item: dict, catalog: dict, rng) -> dict:
    """{real tool name: made-up name} for every tool of the sample's installed apps (experiment E1, training only).

    Each app and each tool gets a new word of two or three syllables, the same throughout the conversation, so the model
    can tell the tools apart only by their descriptions and parameters, as it must for apps it has never seen.
    sys.ask_missing_info keeps its name: it is the same in every sample.
    """
    used, out = set(), {}

    def word() -> str:
        while True:
            w = "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3)))
            if w not in used:
                used.add(w)
                return w

    for app in item["context"]["installed_apps"]:
        if app == "sys":
            continue
        fake = word()
        for name in catalog:
            if name.startswith(app + "."):
                out[name] = f"{fake}.{word()}"
    return out


def renamer(mapping: dict):
    """A function that rewrites every full tool name of ``mapping`` in a text (tool lines, descriptions that point to a
    tool, calls, ask_missing_info's tool_name), or None for an empty mapping."""
    if not mapping:
        return None
    pattern = re.compile(r"(?<![\w.])(" + "|".join(re.escape(n) for n in sorted(mapping, key=len, reverse=True)) + r")(?!\w)")
    return lambda text: pattern.sub(lambda m: mapping[m.group(1)], text)


def think_lead(restated: str, note: str = None) -> str:
    """The first turn's think block after THINK_OPEN: the restated request, then the worked note if there is one."""
    return restated + ("\n" + note if note else "") + THINK_CLOSE


def encode_conversation(item: dict, transcripts: dict, catalog: dict, tok, denoise: bool = False, max_len: int = 0,
                        drop_prob: float = 0.0, rng=None, history_ok=None, rename=None, note: str = None) -> tuple:
    """(ids, labels) of a gold conversation, built segment by segment exactly as the agent loop feeds it.

    ``tok`` maps text to token ids. Labels fall on the assistant turns of the latest request, and on history tool calls
    that ``history_ok`` accepts. Unrelated apps leave the tool list with probability ``drop_prob``, which makes training
    steps cheaper, and then one at a time while the conversation is longer than ``max_len``, so long samples still train
    on a T4. An out-of-scope sample keeps its list: dropping apps there could only make the "no tool" answer easier.
    ``rename`` (from ``renamer``) rewrites every text before it is tokenized. ``note`` (with ``denoise``) is a worked
    note that follows the restated request inside the first think block.
    """
    if rename is not None:
        tok = (lambda t, _tok=tok: _tok(rename(t)))
    removable = apps_to_drop(item) if item["label"]["tool_calls"] else []
    dropped = frozenset(a for a in removable if drop_prob and rng.random() < drop_prob)
    removable = [a for a in removable if a not in dropped]
    while True:
        ids, labels = _encode_once(item, transcripts, catalog, tok, denoise, dropped, history_ok, note)
        if not max_len or len(ids) <= max_len or not removable:
            return ids, labels
        dropped = dropped | {removable.pop(0)}


def _encode_once(item: dict, transcripts: dict, catalog: dict, tok, denoise: bool, dropped: frozenset, history_ok,
                 note: str = None) -> tuple:
    ids, labels = [], []

    def add(text: str, train: bool = False) -> None:
        part = tok(text)
        ids.extend(part)
        labels.extend(part if train else [-100] * len(part))

    add(turn("system", system_text(item, catalog, dropped)))
    for text, call in history_segments(item, transcripts):
        add(text, call is not None and history_ok is not None and history_ok(call))
    add(turn("user", transcripts[item["audio"]]) + (THINK_OPEN if denoise else GEN_PREFIX))
    lead = think_lead(item["audio_text"], note) if denoise else ""
    for call, result in zip(item["label"]["tool_calls"], item["label"]["tool_results"]):
        target = tok(lead + call_text(call["name"], call["arguments"]) + "<|im_end|>")
        lead = ""
        ids += target
        labels += target
        context = tok("\n" + turn("user", result_text(result)) + GEN_PREFIX)
        ids += context
        labels += [-100] * len(context)
    target = tok(lead + DONE + "<|im_end|>")
    ids += target
    labels += target
    return ids, labels


# Distilling step-by-step (8 Nov 2026): the worked note as a training-only side task. A plan example carries this tag
# after the latest request; test-time prompts never do, so the model answers as before and never writes a note there.
PLAN_TAG = "\n[Chỉ viết ghi chú kế hoạch]"
_TOOL_NAME = re.compile(r"(?<![\w.])([a-z][a-z0-9_]*)\.[a-z][a-z0-9_]*")


def note_apps(note: str, catalog: dict) -> set:
    """Apps whose tools the note names (``app.tool``); a plan example never drops them from the tool list."""
    apps = {name.split(".")[0] for name in catalog}
    return {m.group(1) for m in _TOOL_NAME.finditer(note) if m.group(1) in apps}


def encode_plan_example(item: dict, transcripts: dict, catalog: dict, tok, note: str, max_len: int = 0,
                        drop_prob: float = 0.0, rng=None) -> tuple:
    """(ids, labels) of the plan-only side task of one sample (distilling step-by-step).

    The prompt is the sample's own (system turn, history, latest request) with PLAN_TAG after the request; the think
    block opens, the restated request follows without loss, and the loss falls on the worked note only. Nothing comes
    after the note: no call, no result, no end of turn. History calls carry no labels. Unrelated apps leave the tool
    list as in ``encode_conversation``, except apps the note names, so the note never names a tool the prompt lacks.
    """
    keep = note_apps(note, catalog)
    removable = [a for a in apps_to_drop(item) if a not in keep] if item["label"]["tool_calls"] else []
    dropped = frozenset(a for a in removable if drop_prob and rng.random() < drop_prob)
    removable = [a for a in removable if a not in dropped]
    while True:
        ids, labels = _encode_plan_once(item, transcripts, catalog, tok, note, dropped)
        if not max_len or len(ids) <= max_len or not removable:
            return ids, labels
        dropped = dropped | {removable.pop(0)}


def _encode_plan_once(item: dict, transcripts: dict, catalog: dict, tok, note: str, dropped: frozenset) -> tuple:
    ids = list(tok(turn("system", system_text(item, catalog, dropped))))
    for text, _ in history_segments(item, transcripts):
        ids += tok(text)
    ids += tok(turn("user", transcripts[item["audio"]] + PLAN_TAG) + THINK_OPEN)
    labels = [-100] * len(ids)
    # Tokenized as one text, like the first target of encode_conversation; the loss starts where the note's tokens do.
    given = tok(item["audio_text"] + "\n")
    think = tok(item["audio_text"] + "\n" + note)
    if think[:len(given)] == given:
        cut = len(given)
    else:
        cut = next((i for i, (a, b) in enumerate(zip(given, think)) if a != b), min(len(given), len(think)))
    ids += think
    labels += [-100] * cut + list(think[cut:])
    return ids, labels


# Time and number drills (improve_plan.md step 4): short generated exercises as a training-only side task. A pack of
# drills shares one context time; each drill is a user turn (optional data as a tool returned it, the spoken fragment,
# drills.DRILL_TAG naming the arguments) and an assistant turn with the values as compact JSON. Test prompts never
# carry the tag. The system turn holds the same context lines as a real prompt (time, weekday, the next two weeks).
DATA_PREFIX = "Dữ liệu: "


def drill_turns(pack: dict) -> list:
    """(user text, answer text) of every drill of a pack."""
    from .drills import DRILL_TAG, answer_text
    out = []
    for d in pack["drills"]:
        user = (DATA_PREFIX + dumps(d["data"]) + "\n" if d.get("data") else "") + d["text"]
        out.append((user + DRILL_TAG.format(params=", ".join(d["target"])), answer_text(d["target"])))
    return out


def encode_drill_pack(pack: dict, tok) -> tuple:
    """(ids, labels) of one pack: loss on each answer and its end of turn only."""
    now = pack["now"]
    system = "\n".join([SYSTEM, "", "Bối cảnh:", f"- Bây giờ: {now} ({_WEEKDAYS[datetime.fromisoformat(now).weekday()]})",
                        f"- Các ngày tới: {calendar_hint(now)}"])
    ids = list(tok(turn("system", system)))
    labels = [-100] * len(ids)
    for user, answer in drill_turns(pack):
        prompt = list(tok(turn("user", user) + GEN_PREFIX))
        target = list(tok(answer + END))
        ids += prompt + target
        labels += [-100] * len(prompt) + target
    return ids, labels


# Tool-choice drills (improvment_plan3.md T3): one installed-apps context as a real prompt shows it, then several
# requests, each answered with the name of the first tool to call ("none" when no installed app can serve it).
TOOL_TAG = "\n[Bài tập: chỉ ghi tên tool đầu tiên cần gọi]"


def encode_tool_pack(pack: dict, catalog: dict, tok) -> tuple:
    """(ids, labels) of one tool-choice pack: loss on each answer and its end of turn only."""
    ids = list(tok(turn("system", system_text({"context": pack["context"]}, catalog))))
    labels = [-100] * len(ids)
    for d in pack["drills"]:
        prompt = list(tok(turn("user", d["text"] + TOOL_TAG) + GEN_PREFIX))
        target = list(tok(dumps({"tool": d["tool"]}) + END))
        ids += prompt + target
        labels += [-100] * len(prompt) + target
    return ids, labels
