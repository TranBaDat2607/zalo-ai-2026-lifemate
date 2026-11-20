"""Time and number drills: a side task of short generated exercises (improve_plan.md step 4, METHODOLOGY_REVIEW.md
step 2). Each drill is a context time, a spoken fragment (sometimes with data as a tool would return it) and the value
gold would write for one or two arguments. Targets are computed by this code only: no explanation text, no text
written by a model. Training only (``train --drills``); answer-time prompts never carry ``DRILL_TAG``.

Conventions follow the training gold (measured on the fold's training pool, log.md 12 Nov):
- "thứ X" / "thứ X này" = the next such weekday after today (8 of 8 cases); never generated for today's weekday;
- "thứ X tuần sau" = that weekday of the next Monday-start week; generated only where it is also the next such day
  (4 of 4 cases agree on both readings there);
- an hour is always said with its part of day or a day word that fixes it (no bare hours);
- money said without a unit below one thousand is in thousands ("một trăm rưỡi" = 150000, "hai trăm" = 200000).
Numbers are written as the speech model writes them: mostly words, digits before some units (``DIGIT_SHARE``).
"""
import json
import random
import re
import zlib
from datetime import datetime, timedelta

UNITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
WEEKDAY_WORDS = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]
DRILL_TAG = "\n[Bài tập: chỉ ghi giá trị của {params}]"
# share of numbers written in digits before each unit, as the averaged speech model writes the training pool's latest
# requests (measured by scripts/drills/drill_gate.py on outputs/readings/avg5_train.json; the gate checks the drills)
DIGIT_SHARE = {"giờ": 0.20, "phút": 0.58, "nghìn": 0.39, "ngàn": 0.39, "triệu": 0.69, "độ": 0.78, "phần trăm": 0.64,
               "chỗ": 0.28, "tiếng": 0.0, "": 0.1}

# --------------------------------------------------------------------------------------------------- number words


def two_digits(n: int, rng: random.Random, short: bool = False, chuc: bool = False) -> str:
    """0-99 in words. ``short``: the spoken short form without "mươi" ("ba lăm" = 35, "hai mốt" = 21); ``chuc``: an
    exact ten may be said "X chục" ("bốn chục phút", "năm chục ngàn"), never inside a clock time or after "trăm"."""
    if n < 10:
        return UNITS[n]
    t, u = divmod(n, 10)
    if t == 1:
        return "mười" if u == 0 else "mười " + ("lăm" if u == 5 else UNITS[u])
    if u == 0:
        return UNITS[t] + (" chục" if chuc and rng.random() < 0.35 else " mươi")
    last = {1: "mốt", 4: "tư" if rng.random() < 0.5 else "bốn", 5: "lăm"}.get(u, UNITS[u])
    return f"{UNITS[t]} {last}" if short else f"{UNITS[t]} mươi {last}"


def three_digits(n: int, rng: random.Random, half: bool = True) -> str:
    """``half``: X50 may be said "X trăm rưỡi" (not before a unit: "hai trăm rưỡi nghìn" is not said)."""
    h, r = divmod(n, 100)
    if h == 0:
        return two_digits(r, rng)
    if r == 0:
        return f"{UNITS[h]} trăm"
    if r == 50 and half and rng.random() < 0.5:
        return f"{UNITS[h]} trăm rưỡi"
    if r < 10:
        return f"{UNITS[h]} trăm {'linh' if rng.random() < 0.5 else 'lẻ'} {UNITS[r]}"
    return f"{UNITS[h]} trăm {two_digits(r, rng)}"


def say_number(n: int, rng: random.Random, short_ok: bool = True, unit_follows: bool = False) -> str:
    """Any 0 <= n < 10^12 in words, with the usual spoken shortcuts."""
    if n < 100:
        return two_digits(n, rng, short=short_ok and n > 20 and n % 10 and rng.random() < 0.4, chuc=True)
    if n < 1000:
        return three_digits(n, rng, half=not unit_follows)
    thousand = "nghìn" if rng.random() < 0.55 else "ngàn"
    if n < 10**6:
        k, r = divmod(n, 1000)
        if r == 0:
            return f"{say_number(k, rng, short_ok, True)} {thousand}"
        if r == 500 and rng.random() < 0.6:
            return f"{say_number(k, rng, short_ok, True)} {thousand} rưỡi"
        return f"{say_number(k, rng, short_ok, True)} {thousand} {three_digits(r, rng) if r >= 100 else 'không trăm ' + two_digits(r, rng)}"
    m, r = divmod(n, 10**6)
    head = f"{say_number(m, rng, short_ok, True)} triệu"
    if r == 0:
        return head
    if r == 500_000 and rng.random() < 0.7:
        return f"{head} rưỡi"
    if r % 100_000 == 0 and rng.random() < 0.5:  # "hai triệu tư" = 2,400,000
        d = r // 100_000
        return f"{head} {'tư' if d == 4 and rng.random() < 0.5 else UNITS[d]}"
    k, rest = divmod(r, 1000)
    tail = f"{three_digits(k, rng, half=False) if k >= 100 else 'không trăm ' + two_digits(k, rng)} {thousand}" if k else ""
    if rest:
        tail = (tail + " " if tail else "") + three_digits(rest, rng)
    return f"{head} {tail}"


_DIGIT_ACC: dict = {}  # per unit: error diffusion, so a small family still meets its share (reset by generate)


def use_digits(unit: str, rng: random.Random) -> bool:
    """Digits or words before ``unit``: exactly ``DIGIT_SHARE`` of the cases over a run, random phase per unit."""
    acc = _DIGIT_ACC.get(unit)
    acc = (rng.random() if acc is None else acc) + DIGIT_SHARE.get(unit, 0.1)
    _DIGIT_ACC[unit] = acc - (acc >= 1)
    return acc >= 1


def number_text(n: int, unit: str, rng: random.Random, short_ok: bool = True) -> str:
    """``n`` as the speech model writes it before ``unit``: words, or digits with ``DIGIT_SHARE``."""
    if n < 10**6 and use_digits(unit, rng):
        return str(n)
    return say_number(n, rng, short_ok)


_VALUES = {w: i for i, w in enumerate(UNITS)} | {"mốt": 1, "tư": 4, "lăm": 5, "nhăm": 5, "linh": None, "lẻ": None}


def parse_number(text: str) -> int:
    """Independent reader of ``say_number`` / ``number_text`` output (digits allowed), for the gate's round trip."""
    words = text.split()
    if len(words) == 1 and words[0].isdigit():
        return int(words[0])
    if len(words) == 2 and words[1] in ("triệu", "nghìn", "ngàn") and words[0].replace(",", "").isdigit():
        scale = 10**6 if words[1] == "triệu" else 1000
        whole, _, frac = words[0].partition(",")
        return int(whole) * scale + (int(frac) * scale // 10 ** len(frac) if frac else 0)
    total, group, last_big = 0, 0, 1
    i = 0
    while i < len(words):
        w = words[i]
        if w.isdigit():
            group += int(w)
        elif w == "triệu":
            group = group or 1
            total += group * 10**6
            last_big, group = 10**6, 0
        elif w in ("nghìn", "ngàn"):
            group = group or 1
            total += group * 1000
            last_big, group = 1000, 0
        elif w == "trăm":
            group = (group % 100 or 1) * 100 + group // 100 * 100 if group >= 100 else (group or 1) * 100
        elif w == "rưỡi":
            group += last_big // 2 if last_big > 1 and not group else 50 if group % 100 == 0 and group >= 100 else 0
        elif w in ("mươi", "chục"):
            group = group - group % 10 + (group % 10) * 10
        elif w == "mười":
            group += 10
        elif w in ("linh", "lẻ", "không"):
            pass
        elif w in _VALUES:
            v = _VALUES[w]
            prev = words[i - 1] if i else ""
            if prev in _VALUES and prev not in ("linh", "lẻ", "không") and _VALUES[prev] is not None and \
                    prev not in ("mười",) and group % 10 and words[i - 1] not in ("trăm",):
                group = group - group % 10 + (group % 10) * 10 + v  # "ba lăm" = 35
            else:
                group += v
        i += 1
    if words and words[-1] not in ("triệu", "nghìn", "ngàn", "rưỡi") and last_big > 1 and group < last_big and len(words) >= 2 \
            and words[-2] in ("triệu",) and group < 10:
        group *= last_big // 10  # "hai triệu tư"
    return total + group


# ---------------------------------------------------------------------------------------------------- clock times


def part_of_day(h: int) -> str:
    if 4 <= h <= 10:
        return "sáng"
    if 11 <= h <= 12:
        return "trưa"
    if 13 <= h <= 17:
        return "chiều"
    if 18 <= h <= 21:
        return "tối"
    return "đêm"


def say_part(h: int, rng: random.Random) -> str:
    """The part of day a speaker uses for hour h: as ``part_of_day``, but 22 h is mostly "tối" ("mười giờ tối"), 23 h
    "đêm" or "khuya", and 13 h sometimes "trưa"."""
    if h == 22 and rng.random() < 0.7:
        return "tối"
    if h == 23:
        return rng.choice(["đêm", "khuya"])
    if h == 13 and rng.random() < 0.4:
        return "trưa"
    return part_of_day(h)


def spoken_hour(h: int) -> int:
    return 12 if h % 12 == 0 else h % 12


def clock_text(h: int, m: int, rng: random.Random) -> str:
    """The clock reading of h:m without its part of day ("bảy rưỡi", "tám giờ kém mười lăm", "9 giờ 15")."""
    digits = use_digits("giờ", rng)
    num = (lambda x: str(x)) if digits else (lambda x: two_digits(x, rng))
    if m == 0:
        return f"{num(spoken_hour(h))} giờ" + (" đúng" if rng.random() < 0.05 else "")
    if m == 30 and rng.random() < 0.8:
        return f"{num(spoken_hour(h))}{' giờ' if rng.random() < 0.3 else ''} rưỡi"
    if m >= 40 and m % 5 == 0 and rng.random() < 0.6:
        return f"{num(spoken_hour(h + 1))} giờ kém {num(60 - m)}{' phút' if rng.random() < 0.2 else ''}"
    return f"{num(spoken_hour(h))} giờ {num(m)}{' phút' if rng.random() < 0.25 else ''}"


def parse_clock(text: str, part: str) -> tuple:
    """Independent reader of ``clock_text`` + part of day -> (hour 0-23, minute)."""
    t = text.replace(" đúng", "").replace(" phút", "")
    if " kém " in t:
        hh, mm = t.split(" kém ")
        h, m = parse_number(hh.replace(" giờ", "")) - 1, 60 - parse_number(mm)
    elif t.endswith("rưỡi"):
        h, m = parse_number(t.replace(" rưỡi", "").replace(" giờ", "")), 30
    else:
        hh, _, mm = t.partition(" giờ")
        h, m = parse_number(hh), parse_number(mm.strip()) if mm.strip() else 0
    h %= 12
    if part in ("chiều", "tối") or (part == "trưa" and h < 11 and h != 0) or (part in ("đêm", "khuya") and h >= 9):
        h += 12
    if part == "trưa" and h == 0:
        h = 12
    return h % 24, m


# ---------------------------------------------------------------------------------------------------- the drills

PRONOUNS = [("em", 385), ("anh", 246), ("tôi", 239), ("chị", 198), ("mình", 84), ("con", 81), ("cô", 78), ("chú", 43),
            ("tớ", 25), ("tui", 15)]
ENDINGS = {"north": ["nhé", "nhé", "với", "nhé", "nhé", "giúp {p} với", "hộ {p} nhé", "nhé", "", "", "với nhé"],
           "south": ["nha", "nha", "nha", "nha", "giùm {p} nha", "nghe", "nghen", "", "giùm {p}", "hen"],
           "central": ["nghe", "nghe", "nha", "", "giùm {p} nghe", "giùm {p} với"]}
# the helper word a region uses in a frame's "giúp" (measured shares in the pool's readings: giúp .21, giùm .15)
HELPER = {"north": [("giúp", 8), ("hộ", 1)], "south": [("giùm", 5), ("giúp", 4), ("dùm", 1)], "central": [("giùm", 5), ("giúp", 5)]}
REGION_SHARE = [("north", 0.48), ("south", 0.44), ("central", 0.08)]
TODAY_WORDS = {"north": ["nay", "hôm nay"], "south": ["nay", "bữa nay", "hôm nay"], "central": ["ni", "nay", "hôm ni"]}


def _pick(rng, weighted):
    total = sum(w for _, w in weighted)
    x = rng.random() * total
    for item, w in weighted:
        x -= w
        if x <= 0:
            return item
    return weighted[-1][0]


class Ctx:
    def __init__(self, now: datetime, rng: random.Random):
        self.now, self.rng = now, rng
        self.region = _pick(rng, REGION_SHARE)
        self.who = _pick(rng, PRONOUNS)

    def end(self, frame: str = "") -> str:
        options = ENDINGS[self.region]
        if any(w in frame for w in ("giúp", "giùm", "dùm", "hộ")):
            options = [o for o in options if not any(w in o for w in ("giúp", "giùm", "dùm", "hộ"))] or [""]
        e = self.rng.choice(options).format(p=self.who)
        return (" " + e) if e else ""

    def today(self) -> str:
        return self.rng.choice(TODAY_WORDS[self.region])


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:00+07:00")


def day_phrase(ctx: Ctx, day: datetime, part: str):
    """(words for that day with the part of day, form) or None when no unambiguous phrase exists."""
    rng, now = ctx.rng, ctx.now
    delta = (day.date() - now.date()).days
    if delta == 0:
        t = ctx.today()
        return (f"{part} {t.split()[-1]}" if t in ("nay", "ni") or rng.random() < 0.6 else f"{part} {t}"), "today"
    if delta == 1:
        return rng.choice([f"{part} mai", f"{part} mai", f"{part} ngày mai"]), "tomorrow"
    if delta == 2:
        return rng.choice([f"{part} mốt", f"{part} ngày kia", f"{part} ngày mốt"]), "day after"
    if delta == -1:
        return f"{part} qua" if part in ("sáng", "trưa", "chiều", "tối") else None, "yesterday"
    wd = day.weekday()
    if 3 <= delta <= 6 and wd != 6:  # bare "chủ nhật" is left out: train gold reads it both ways (train_0701)
        return f"{part} {WEEKDAY_WORDS[wd]}{rng.choice(['', ' này', ' tuần này']) if delta < 7 - now.weekday() else ''}", "weekday"
    if 7 <= delta <= 13 and wd < now.weekday() and delta == (wd - now.weekday()) % 7 + 7 - 7 * (wd >= now.weekday()):
        return f"{part} {WEEKDAY_WORDS[wd]} tuần {rng.choice(['sau', 'tới', 'sau'])}", "next week"
    return None


def say_day(d: int, rng: random.Random) -> str:
    """A day of the month: "mười bảy", "hai mốt", "hai mươi hai", sometimes digits."""
    if use_digits("", rng):
        return str(d)
    return two_digits(d, rng, short=d > 20 and d % 10 == 1 and rng.random() < 0.6)


def far_day_phrase(ctx: Ctx, day: datetime, part: str):
    """A day three or more days ahead said by its date: "sáng hôm mười bảy", "chiều ngày hai mốt tháng mười"."""
    rng, now = ctx.rng, ctx.now
    if (day.date() - now.date()).days < 3:
        return None
    dn = say_day(day.day, rng)
    if day.month == now.month and day.year == now.year:
        said = rng.choice([f"hôm {dn}", f"ngày {dn}", f"ngày {dn} tháng này", f"{'bữa' if ctx.region == 'south' else 'hôm'} {dn}"])
        form = "day of month"
    else:
        said = rng.choice([f"ngày {dn} tháng {two_digits(day.month, rng)}", f"{dn} tháng {two_digits(day.month, rng)}",
                           f"hôm {dn} tháng {two_digits(day.month, rng)}"])
        if day.month == (now.month % 12) + 1 and rng.random() < 0.3:
            said = f"ngày {dn} tháng sau"
        form = "day+month"
    return f"{part} {said}", form


def drill_rule(ctx: Ctx) -> dict:
    """A daily device rule: gold keeps HH:MM ("cứ mười giờ tối là bật đèn" -> 22:00)."""
    rng = ctx.rng
    h = _pick(rng, [(x, 3 if 17 <= x <= 23 else 2 if 5 <= x <= 8 else 1) for x in range(0, 24)])
    m = _pick(rng, [(0, 8), (30, 3), (15, 1), (45, 1)])
    part = say_part(h, rng)
    clock = clock_text(h, m, rng)
    assert parse_clock(clock, part) == (h, m), (clock, part, h, m)
    dev = rng.choice(["đèn phòng khách", "đèn cổng", "đèn biển hiệu", "máy lọc không khí", "quạt phòng ngủ", "đèn ban công",
                      "ổ cắm máy nước nóng", "rèm phòng khách"])
    act = rng.choice(["bật", "tắt"])
    text = rng.choice([f"Cài tự động đi, cứ {clock} {part} là {act} {dev}", f"Cài cho {ctx.who} cái hẹn giờ, {part} nào {clock} cũng tự {act} {dev}",
                       f"Hằng ngày {clock} {part} tự {act} {dev} giúp {ctx.who}", f"Tạo quy tắc {act} {dev} lúc {clock} {part} mỗi ngày"])
    return {"family": "clock", "form": f"rule/{_form(clock)}", "text": text + ctx.end(text), "target": {"time": f"{h:02d}:{m:02d}"},
            "checks": [("clock", clock, part, h, m)]}


def drill_clock(ctx: Ctx) -> dict:
    """A clock time with its part of day and day, relative to now -> ISO date-time (or HH:MM for device schedules)."""
    rng, now = ctx.rng, ctx.now
    for _ in range(50):
        delta = _pick(rng, [(0, 36), (1, 40), (2, 5), (3, 3), (4, 3), (5, 2), (6, 2), (8, 2), (10, 1), (12, 1)] +
                      [(x, 0.3) for x in range(7, 41)])
        h = _pick(rng, [(x, 3 if 6 <= x <= 22 else 1) for x in range(4, 24)])
        m = _pick(rng, [(0, 50), (30, 25), (15, 7), (45, 6), (10, 3), (20, 3), (40, 2), (50, 2), (5, 1), (35, 1)])
        day = (now + timedelta(days=delta)).replace(hour=h, minute=m, second=0)
        if day <= now:
            continue
        part = say_part(h, rng)
        dp = day_phrase(ctx, day, part)
        if delta >= 3 and (dp is None or dp[0] is None or rng.random() < 0.4):
            dp = far_day_phrase(ctx, day, part)
        if dp is None or dp[0] is None:
            continue
        clock = clock_text(h, m, rng)
        assert parse_clock(clock, part) == (h, m), (clock, part, h, m)
        order = rng.random()
        said = f"{clock} {dp[0]}" if order < 0.55 else f"{dp[0]} {clock}" if order < 0.85 else f"{dp[0]} lúc {clock}"
        frame = rng.choice(["Nhắc {w} {t}{e}", "Đặt báo thức {t} giúp {w}{e}", "{T} nhắc {w} gọi lại cho khách{e}",
                            "Hẹn {w} {t} đi đổ xăng{e}", "Tạo lịch họp nhóm {t}{e}", "Đặt xe đón {w} {t}{e}"])
        param = "time" if "báo thức" in frame or "Nhắc" in frame or "nhắc" in frame else "start"
        if "Đặt xe" in frame:
            param = "schedule_time"
        text = frame.format(w=ctx.who, t=said, T=said[0].upper() + said[1:], e=ctx.end())
        value = iso(day)
        return {"family": "clock", "form": f"{dp[1]}/{_form(clock)}", "text": text, "target": {param: value},
                "checks": [("clock", clock, part, h, m)]}
    raise RuntimeError("no clock drill")


def _form(clock: str) -> str:
    if "kém" in clock:
        return "kém"
    if "rưỡi" in clock:
        return "rưỡi"
    if any(c.isdigit() for c in clock):
        return "digits"
    return "giờ+phút" if clock.count(" ") >= 2 else "giờ"


def drill_relative(ctx: Ctx) -> dict:
    """"X phút nữa", "một tiếng rưỡi nữa" -> now + offset."""
    rng, now = ctx.rng, ctx.now
    minutes = _pick(rng, [(10, 2), (15, 4), (20, 2), (30, 5), (45, 3), (60, 5), (90, 3), (120, 3), (150, 1), (180, 1)])
    said, checks = duration_text(minutes, rng)
    text = rng.choice([f"{said} nữa nhắc {ctx.who} tắt bếp{ctx.end()}", f"Nhắc {ctx.who} sau {said} nữa ra lấy đồ{ctx.end()}",
                       f"Đặt báo thức {said} nữa{ctx.end()}"])
    return {"family": "clock", "form": "relative/" + _dur_form(said), "text": text[0].upper() + text[1:],
            "target": {"time": iso(now + timedelta(minutes=minutes))}, "checks": checks}


def duration_text(minutes: int, rng: random.Random) -> tuple:
    """Words for a duration: "bốn lăm phút", "một tiếng rưỡi", "nửa tiếng", "hai tiếng mười lăm phút"."""
    h, m = divmod(minutes, 60)
    if minutes == 30 and rng.random() < 0.7:
        return "nửa tiếng", [("dur", "nửa tiếng", 30)]
    if h and m == 30 and rng.random() < 0.7:
        s = f"{number_text(h, 'tiếng', rng)} tiếng rưỡi" if h > 1 or rng.random() < 0.5 else "tiếng rưỡi"
        return s, [("dur", s, minutes)]
    if h and m == 0 and rng.random() < 0.8:
        s = f"{number_text(h, 'tiếng', rng)} tiếng"
        return s, [("dur", s, minutes)]
    if h and rng.random() < 0.5:
        s = f"{number_text(h, 'tiếng', rng)} tiếng {number_text(m, 'phút', rng)} phút"
        return s, [("dur", s, minutes)]
    s = f"{number_text(minutes, 'phút', rng)} phút"
    return s, [("dur", s, minutes)]


def parse_duration(text: str) -> int:
    t = text.replace(" nữa", "")
    if t == "nửa tiếng":
        return 30
    if t == "tiếng rưỡi":
        return 90
    total = 0
    if " tiếng" in t:
        hh, _, rest = t.partition(" tiếng")
        total += parse_number(hh) * 60
        rest = rest.strip()
        if rest == "rưỡi":
            total += 30
        elif rest:
            total += parse_number(rest.replace(" phút", ""))
        return total
    return parse_number(t.replace(" phút", ""))


def _dur_form(s: str) -> str:
    if "nửa" in s:
        return "nửa tiếng"
    if "rưỡi" in s:
        return "tiếng rưỡi"
    if "tiếng" in s and "phút" in s:
        return "tiếng+phút"
    if "tiếng" in s:
        return "tiếng"
    return "digits" if any(c.isdigit() for c in s) else "phút"


def drill_span(ctx: Ctx) -> dict:
    """Minutes between two clock times, from a clock time until now, or of a stated duration. The frames are everyday
    ones (timers, parking, meetings, appliances, time worked) and none comes from an app held out of validation."""
    rng, now, who = ctx.rng, ctx.now, ctx.who
    kind = _pick(rng, [("between", 40), ("until now", 15), ("duration", 45)])
    if kind == "duration":
        param = _pick(rng, [("minutes", 45), ("delay_minutes", 25), ("hours", 15), ("duration", 15)])
        if param == "hours":
            hours = _pick(rng, [(1, 3), (2, 4), (3, 3), (4, 2), (5, 1), (6, 1), (8, 1)])
            said = rng.choice([f"{say_number(hours, rng)} tiếng", f"{say_number(hours, rng)} tiếng",
                               f"{say_number(hours, rng)} tiếng đồng hồ"] + ([f"{say_number(hours, rng)} giờ"] if hours > 1 else []))
            text = rng.choice([f"Đặt chỗ gửi xe {said} giúp {who}", f"Giữ chỗ đậu xe cho {who} {said}",
                               f"{who.capitalize()} cần gửi xe khoảng {said}, đặt chỗ đi"])
            return {"family": "span", "form": "duration/hours", "text": text + ctx.end(text), "target": {"hours": hours},
                    "checks": []}
        minutes = _pick(rng, [(x, 1) for x in (5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 75, 90, 120, 150, 180)])
        said, checks = duration_text(minutes, rng)
        if param == "delay_minutes":
            app = rng.choice(["máy giặt", "máy rửa chén", "máy sấy", "nồi cơm", "máy giặt"])
            text = rng.choice([f"Hẹn {app} {said} nữa mới chạy", f"Cho {app} chạy sau {said} nữa",
                               f"{said.capitalize()} nữa thì bật {app} giúp {who}", f"Hẹn giờ {app} chạy sau {said}"])
        elif param == "duration":
            text = ""
        else:
            text = rng.choice([f"Ghi lại hôm nay {who} làm việc được {said}", f"Ghi giúp {who} thời gian dọn nhà {said}",
                               f"{who.capitalize()} vừa nấu ăn mất {said}, ghi lại", f"Ghi thời gian {who} lái xe {said}"])
        target = {"delay_minutes": minutes} if param == "delay_minutes" else {"minutes": minutes}
        if param == "duration":  # a meeting of that length starting at a stated clock time
            h = _pick(rng, [(x, 1) for x in range(8, 20)])
            start = (now + timedelta(days=1)).replace(hour=h, minute=0, second=0)
            part = part_of_day(h)
            c1 = clock_text(h, 0, rng)
            text = rng.choice([f"Tạo cuộc họp lúc {c1} {part} mai, dài {said} giúp {who}",
                               f"{c1.capitalize()} {part} mai có cuộc gọi, chắc kéo dài {said}, đặt lịch giúp {who}",
                               f"Đặt lịch hẹn {part} mai lúc {c1}, kéo dài {said}"])
            checks = checks + [("clock", c1, part, h, 0)]
            target = {"start": iso(start), "end": iso(start + timedelta(minutes=minutes))}
        return {"family": "span", "form": f"duration/{param}/" + _dur_form(said), "text": text + ctx.end(text),
                "target": target, "checks": checks}
    for _ in range(50):
        if kind == "until now":
            minutes = _pick(rng, [(x, 1) for x in (15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90, 120)])
            start = now - timedelta(minutes=minutes)
            if start.date() != now.date() or start.minute % 5 or (start.minute not in (0, 30) and rng.random() < 0.6):
                continue
            part = part_of_day(start.hour)
            c1 = clock_text(start.hour, start.minute, rng)
            assert parse_clock(c1, part) == (start.hour, start.minute)
            until = rng.choice(["đến giờ", "tới giờ", "tới bây giờ", "đến bây giờ"])
            text = rng.choice([f"{who.capitalize()} làm việc từ {c1} {until}, ghi lại số phút",
                               f"Ghi thời gian dọn nhà từ {c1} {part} {ctx.today()} {until}",
                               f"Từ {c1} {until} {who} lái xe liên tục, ghi lại giúp {who}"])
            return {"family": "span", "form": "until now/" + _form(c1), "text": text + ctx.end(text),
                    "target": {"minutes": minutes}, "checks": [("clock", c1, part, start.hour, start.minute)]}
        h1 = _pick(rng, [(x, 1) for x in range(6, 23)])
        m1 = _pick(rng, [(0, 4), (30, 4), (15, 2), (45, 1), (10, 1), (20, 1)])
        minutes = _pick(rng, [(x, 1) for x in (20, 25, 30, 35, 40, 45, 50, 60, 75, 90, 120, 150, 180)])
        a = now.replace(hour=h1, minute=m1, second=0)
        b = a + timedelta(minutes=minutes)
        if b.date() != a.date():
            continue
        part, p2 = part_of_day(h1), part_of_day(b.hour)
        if p2 != part:
            continue
        c1, c2 = clock_text(a.hour, a.minute, rng), clock_text(b.hour, b.minute, rng)
        assert parse_clock(c1, part) == (a.hour, a.minute) and parse_clock(c2, part) == (b.hour, b.minute)
        to = rng.choice(["tới", "đến"])
        where = rng.random()
        if rng.random() < 0.5:  # a calendar entry: start and end on a day
            day = now + timedelta(days=_pick(rng, [(0, 3), (1, 4), (2, 2)]))
            a, b = day.replace(hour=a.hour, minute=a.minute, second=0), day.replace(hour=b.hour, minute=b.minute, second=0)
            got = day_phrase(ctx, day, part)
            if not got or a <= now:
                continue
            words = got[0]
            span = f"{words} từ {c1} {to} {c2}" if where < 0.6 else f"từ {c1} {to} {c2} {words}"
            text = rng.choice([f"Đặt lịch họp {span}", f"Tạo lịch {span} đi khám bệnh", f"Thêm vào lịch {span} gặp khách",
                               f"Xếp lịch cà phê với bạn {span} giúp {who}"])
            return {"family": "span", "form": f"between/calendar/{_form(c1)}+{_form(c2)}", "text": text + ctx.end(text),
                    "target": {"start": iso(a), "end": iso(b)},
                    "checks": [("clock", c1, part, a.hour, a.minute), ("clock", c2, part, b.hour, b.minute)]}
        span = (f"từ {c1} {to} {c2} {part}" if where < 0.5 else f"{part} {ctx.today()} từ {c1} {to} {c2}" if where < 0.8
                else f"từ {c1} {part} {to} {c2}")
        if rng.random() < 0.12:  # a corrected start, as in "từ chín giờ, à nhầm, chín rưỡi đến ..."
            wrong = clock_text(a.hour, 0 if a.minute else 30, rng)
            if wrong != c1:
                span = f"từ {wrong}, {rng.choice(['à nhầm', 'à không'])}, {c1} {to} {c2}"
        text = rng.choice([f"{who.capitalize()} làm việc {span}, ghi lại giúp {who}", f"Ghi thời gian dọn nhà {span}",
                           f"Ghi giúp {who} {span} {who} ngồi lái xe", f"Tính xem {span} là bao nhiêu phút rồi ghi lại"])
        return {"family": "span", "form": f"between/{_form(c1)}+{_form(c2)}", "text": text + ctx.end(text),
                "target": {"minutes": minutes},
                "checks": [("clock", c1, part, a.hour, a.minute), ("clock", c2, part, b.hour, b.minute)]}
    raise RuntimeError("no span drill")


def _event(title: str, start: datetime, end: datetime, kind: str = "events") -> dict:
    return {"id": f"ev_{zlib.crc32(f'{title}{start}'.encode()) % 900 + 100}", "title": title, "start": iso(start), "end": iso(end)}


def drill_from_data(ctx: Ctx) -> dict:
    """Spans and offsets that need times from a read result: the length of an event, the gap between two events, a
    move of an event, a time before an event, minutes from now until an event."""
    rng, now = ctx.rng, ctx.now
    day = now + timedelta(days=_pick(rng, [(0, 5), (1, 5)]))
    h = _pick(rng, [(x, 1) for x in range(7, 21)])
    m = _pick(rng, [(0, 6), (30, 4), (15, 1), (45, 1)])
    start = day.replace(hour=h, minute=m, second=0)
    length = _pick(rng, [(30, 2), (45, 2), (60, 5), (75, 1), (90, 2), (120, 2)])
    end = start + timedelta(minutes=length)
    title = rng.choice(["Họp team", "Cà phê với gia đình", "Đi siêu thị", "Khám răng", "Họp phụ huynh", "Ăn tối với bạn", "Đi chơi công viên",
                        "Phỏng vấn", "Họp kế hoạch tháng"])
    ev = _event(title, start, end)
    kind = _pick(rng, [("length", 15), ("gap", 15), ("move", 25), ("before", 25), ("until", 10), ("move+before", 10)])
    data = {"ok": True, "events": [ev]}
    if kind == "length":
        text = rng.choice([f"Lịch {title.lower()} dài bao lâu thì ghi thời gian làm việc bấy nhiêu{ctx.end()}",
                           f"Ghi thời gian {ctx.who} lái xe đúng bằng thời lượng buổi {title.lower()}{ctx.end()}"])
        return {"family": "span", "form": "data/length", "text": text, "data": data, "target": {"minutes": length},
                "checks": []}
    if kind == "gap":
        gap = _pick(rng, [(15, 1), (30, 3), (45, 2), (60, 3), (90, 1), (120, 1)])
        s2 = end + timedelta(minutes=gap)
        e2 = s2 + timedelta(minutes=60)
        if s2.date() != end.date():
            gap, s2, e2 = 30, end + timedelta(minutes=30), end + timedelta(minutes=90)
        data["events"].append(_event(rng.choice(["Đón con", "Họp khách", "Ăn trưa", "Gọi điện"]), s2, e2))
        text = rng.choice([f"Ghi thời gian dọn nhà đúng bằng khoảng trống giữa hai lịch{ctx.end()}",
                           f"Đặt lịch gọi điện cho mẹ vào đúng khoảng trống giữa hai lịch đó{ctx.end()}"])
        target = {"minutes": gap} if "Ghi" in text else {"start": iso(end), "end": iso(s2)}
        return {"family": "span", "form": "data/gap", "text": text, "data": data, "target": target, "checks": []}
    if kind == "until":
        if start <= now or start.date() != now.date():
            start = now.replace(second=0) + timedelta(minutes=_pick(rng, [(60, 1), (85, 1), (120, 1), (150, 1)]))
            start = start.replace(minute=(start.minute // 5) * 5)
            ev = _event(title, start, start + timedelta(minutes=60))
            data = {"ok": True, "events": [ev]}
        minutes = int((start - now.replace(second=0)).total_seconds() // 60)
        text = rng.choice([f"Hẹn máy giặt chạy đúng giờ {title.lower()} luôn{ctx.end()}",
                           f"Đợi tới giờ {title.lower()} hẵng chạy máy rửa chén{ctx.end()}"])
        return {"family": "chain", "form": "data/delay", "text": text, "data": data, "target": {"delay_minutes": minutes},
                "checks": []}
    shift = _pick(rng, [(15, 2), (30, 4), (45, 2), (60, 5), (90, 2), (120, 3)])
    shift_text, checks = duration_text(shift, rng)
    if kind in ("move", "move+before"):
        later = rng.random() < 0.6
        word = rng.choice(["muộn thêm", "trễ", "lùi", "muộn hơn"]) if later else rng.choice(["sớm hơn", "sớm lên", "lên sớm"])
        sign = 1 if later else -1
        ns, ne = start + sign * timedelta(minutes=shift), end + sign * timedelta(minutes=shift)
        text = f"Dời {'giúp ' + ctx.who + ' ' if rng.random() < 0.45 else ''}lịch {title.lower()} {word} {shift_text}"
        target = {"start": iso(ns), "end": iso(ne)}
        form = "data/move"
        if kind == "move+before":
            before = _pick(rng, [(15, 2), (30, 2), (60, 3)])
            b_text, b_checks = duration_text(before, rng)
            checks = checks + b_checks
            text += f", đặt báo thức trước đó {b_text}"
            target = {"time": iso(ns - timedelta(minutes=before))}
            form = "data/move+before"
        return {"family": "chain", "form": form, "text": text + ctx.end(), "data": data, "target": target, "checks": checks}
    word = rng.choice(["trước giờ đó", "trước giờ hẹn", "sớm hơn giờ hẹn", "trước"])
    text = rng.choice([f"Đặt xe đón {ctx.who} {word} {shift_text}{ctx.end()}", f"Đặt giúp {ctx.who} xe đón {word} {shift_text}{ctx.end()}",
                       f"Nhắc {ctx.who} {word} {shift_text}{ctx.end()}"])
    param = "schedule_time" if "xe" in text else "time"
    return {"family": "chain", "form": "data/before", "text": text, "data": data,
            "target": {param: iso(start - timedelta(minutes=shift))}, "checks": checks}


def drill_date(ctx: Ctx) -> dict:
    """A day said relative to today -> YYYY-MM-DD."""
    rng, now = ctx.rng, ctx.now
    for _ in range(50):
        kind = _pick(rng, [("rel", 45), ("weekday", 30), ("next week", 10), ("day of month", 15)])
        if kind == "rel":
            delta = _pick(rng, [(0, 3), (1, 6), (2, 2), (-1, 2)])
            words = {0: [ctx.today(), "hôm nay"], 1: ["mai", "ngày mai"], 2: ["mốt", "ngày kia", "ngày mốt"],
                     -1: ["hôm qua", "bữa qua" if ctx.region == "south" else "hôm qua"]}[delta]
            said, form = rng.choice(words), {0: "today", 1: "tomorrow", 2: "day after", -1: "yesterday"}[delta]
        elif kind in ("weekday", "next week"):
            wd = rng.randrange(7)
            if kind == "weekday":
                delta = (wd - now.weekday()) % 7
                if delta == 0 or wd == 6:  # bare "chủ nhật": train gold reads it both ways (train_0701)
                    continue
                said = WEEKDAY_WORDS[wd] + (rng.choice(["", " này"]) if delta < 7 - now.weekday() or wd == 6 else "")
                if delta >= 7 - now.weekday() and wd != 6:  # past this week's Sunday: only the bare weekday
                    said = WEEKDAY_WORDS[wd]
                form = "weekday"
            else:
                if wd >= now.weekday():
                    continue
                delta = (wd - now.weekday()) % 7
                said, form = f"{WEEKDAY_WORDS[wd]} tuần {rng.choice(['sau', 'tới'])}", "next week"
        else:
            delta = rng.randrange(1, 20)
            d = now + timedelta(days=delta)
            if d.month != now.month:
                said = f"ngày {number_text(d.day, '', rng, short_ok=False)} tháng {number_text(d.month, '', rng, short_ok=False)}"
                form = "day+month"
            else:
                said, form = f"{rng.choice(['ngày', 'hôm', 'bữa' if ctx.region == 'south' else 'ngày'])} {number_text(d.day, '', rng, short_ok=False)}", "day of month"
        d = (now + timedelta(days=delta)).date()
        text = rng.choice([f"{said[0].upper() + said[1:]} {ctx.who} có lịch gì không", f"Xem giúp {ctx.who} {said} có mưa không{ctx.end()}",
                           f"Coi {said} trời sao{ctx.end()}", f"Dò vé số xổ {said} giùm {ctx.who}{ctx.end()}" if ctx.region == "south"
                           else f"Kiểm tra lịch {said} giúp {ctx.who}{ctx.end()}"])
        return {"family": "date", "form": form, "text": text, "target": {"date": d.isoformat()}, "checks": []}
    raise RuntimeError("no date drill")


def drill_amount(ctx: Ctx) -> dict:
    """Money and counts: spoken amounts, percentages, degrees, seats, shopping quantities, foreign
    currency, savings terms and simple arithmetic on a read value."""
    rng = ctx.rng
    kinds = [("money", 38), ("money bare", 8), ("percent", 9), ("degrees", 9), ("count", 7), ("quantity", 8),
             ("currency", 15), ("savings", 6), ("arith", 8)]
    if _AMOUNT_ONLY:
        kinds = [(k, w) for k, w in kinds if k in _AMOUNT_ONLY]
    kind = _pick(rng, kinds)
    if kind == "quantity":
        v = _pick(rng, [(1, 3), (2, 5), (3, 4), (4, 2), (5, 2), (6, 1), (10, 1), (12, 1)])
        item = rng.choice(["hộp sữa", "chai nước mắm", "gói mì", "cân táo", "kí thịt", "bịch khăn giấy", "lốc nước suối",
                           "hộp trứng", "chai dầu ăn", "túi gạo"])
        said = number_text(v, "", rng)
        text = rng.choice([f"Thêm {said} {item} vào danh sách cần mua", f"Ghi vào danh sách đi chợ {said} {item} cho {ctx.who}",
                           f"Mua thêm {said} {item}, ghi vào danh sách"])
        return {"family": "amount", "form": "quantity", "text": text + ctx.end(text), "target": {"quantity": v},
                "checks": [("number", said, v)]}
    if kind == "currency":  # a foreign amount keeps its own unit: "năm trăm đô" = 500
        v = _pick(rng, [(x, 1) for x in (50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 4500, 80000)])
        cur = rng.choice(["đô", "đô Mỹ", "ơ-rô", "yên"] if v < 10000 else ["yên", "won"])
        said = say_number(v, rng, unit_follows=True)
        if v >= 1000 and use_digits("nghìn", rng):
            said = f"{v / 1000:g} {rng.choice(['nghìn', 'ngàn'])}".replace(".", ",")
        text = rng.choice([f"Đổi {said} {cur} ra tiền Việt được bao nhiêu", f"Tính hộ {ctx.who} {said} {cur} ra tiền Việt",
                           f"{said.capitalize()} {cur} quy ra tiền Việt là bao nhiêu"])
        return {"family": "amount", "form": "currency", "text": text + ctx.end(text), "target": {"amount": v},
                "checks": [("number", said, v)]}
    if kind == "savings":
        value = _pick(rng, [(x * 10**6, 1) for x in (5, 10, 20, 30, 40, 50, 100)])
        months = _pick(rng, [(1, 1), (3, 3), (6, 3), (9, 1), (12, 3), (24, 1)])
        said = number_text(value // 10**6, "triệu", rng) + " triệu"
        ms = number_text(months, "", rng)
        text = rng.choice([f"Gửi tiết kiệm {said} kỳ hạn {ms} tháng", f"Mở sổ tiết kiệm {ms} tháng, gửi {said}",
                           f"Gửi {said} tiết kiệm {ms} tháng từ tài khoản thanh toán"])
        return {"family": "amount", "form": "savings", "text": text + ctx.end(text),
                "target": {"amount": value, "term_months": months}, "checks": [("number", said, value), ("number", ms, months)]}
    if kind in ("money", "money bare"):
        if kind == "money bare":  # below a thousand without unit = thousands (gold: "một trăm rưỡi" = 150000)
            k = _pick(rng, [(x, 1) for x in (100, 150, 200, 250, 300, 500)])
            said = three_digits(k, rng)
            value = k * 1000
            form = "bare hundreds"
            text = rng.choice([f"Chuyển cho Huy {said} qua ví{ctx.end()}", f"Mỗi tờ trúng thì chuyển nó {said}{ctx.end()}",
                               f"Chuyển ví anh Long {said}{ctx.end()}"])
            return {"family": "amount", "form": form, "text": text, "target": {"amount": value}, "checks": [("number", said, k)]}
        else:
            value = _pick(rng, [(x, 1) for x in (20_000, 30_000, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 500_000,
                                                  1_000_000, 1_500_000, 2_000_000, 2_400_000, 2_500_000, 3_000_000, 3_500_000, 5_000_000, 12_500_000,
                                                  40_000_000, 35_000, 45_000, 120_000)])
            unit = "triệu" if value >= 10**6 else "nghìn"
            if use_digits(unit, rng):
                said = (f"{value // 10**6} triệu" if value % 10**6 == 0 else f"{value / 10**6:g} triệu".replace(".", ",")) \
                    if value >= 10**6 else f"{value // 1000} {rng.choice(['nghìn', 'ngàn'])}"
            else:
                said = say_number(value, rng)
            form = ("triệu rưỡi" if "triệu rưỡi" in said else "triệu" if "triệu" in said else
                    "chục nghìn" if "chục" in said else "digits" if any(c.isdigit() for c in said) else "nghìn")
        text = rng.choice([f"Chuyển cho Lan {said} qua ví{ctx.end()}", f"Chuyển giúp {ctx.who} {said} cho Lan qua ví{ctx.end()}", f"Nạp {said} tiền điện thoại cho {ctx.who}{ctx.end()}",
                           f"Chuyển {said} từ tài khoản thanh toán{ctx.end()}", f"Gửi tiết kiệm {said}{ctx.end()}"])
        return {"family": "amount", "form": form, "text": text, "target": {"amount": value}, "checks": [("number", said, value)]}
    if kind == "percent":
        v = _pick(rng, [(10, 1), (20, 1), (30, 2), (40, 2), (50, 2), (60, 1), (70, 1), (80, 1)])
        if v == 50 and rng.random() < 0.3:
            said, checks = "một nửa", []
        else:
            said = number_text(v, "phần trăm", rng) + " phần trăm"
            checks = [("number", said.replace(" phần trăm", ""), v)]
        text = f"Giảm độ sáng đèn phòng ngủ xuống còn {said}{ctx.end()}"
        return {"family": "amount", "form": "percent", "text": text, "target": {"brightness": v}, "checks": checks}
    if kind == "degrees":
        v = rng.randrange(18, 30)
        said = number_text(v, "độ", rng)
        text = f"Bật điều hoà phòng khách {said} độ{ctx.end()}"
        return {"family": "amount", "form": "degrees", "text": text, "target": {"temperature": v}, "checks": [("number", said, v)]}
    if kind == "count":
        v = rng.randrange(1, 6)
        said = number_text(v, "chỗ", rng)
        text = f"Đặt cho {ctx.who} {said} chỗ chuyến sớm nhất{ctx.end()}"
        return {"family": "amount", "form": "seats", "text": text, "target": {"passengers": v}, "checks": [("number", said, v)]}
    base = _pick(rng, [(x, 1) for x in (200, 250, 300, 350, 400)])
    d = _pick(rng, [(x, 1) for x in (20, 50, 100)])
    down = rng.random() < 0.6
    said = say_number(d, rng)
    text = f"Lấy mức cảnh báo điện đang để {'trừ' if down else 'cộng'} {said} số{ctx.end()}"
    data = {"ok": True, "threshold_kwh": base}
    if rng.random() < 0.4:
        t0 = rng.randrange(24, 29)
        dt = rng.randrange(1, 4)
        said = number_text(dt, "độ", rng)
        text = f"Hạ điều hoà phòng ngủ xuống {said} độ{ctx.end()}"
        return {"family": "amount", "form": "arith/degrees", "text": text, "data": {"ok": True, "temperature": t0},
                "target": {"temperature": t0 - dt}, "checks": [("number", said, dt)]}
    return {"family": "amount", "form": "arith", "text": text, "data": data,
            "target": {"threshold_kwh": base - d if down else base + d}, "checks": [("number", said, d)]}


_AMOUNT_ONLY = None
MONEY_KINDS = {"money", "money bare", "currency", "savings"}


def _only(kinds):
    def fn(ctx):
        global _AMOUNT_ONLY
        _AMOUNT_ONLY = kinds
        try:
            return drill_amount(ctx)
        finally:
            _AMOUNT_ONLY = None
    fn.__name__ = "drill_amount_" + ("money" if kinds == MONEY_KINDS else "number")
    return fn


drill_money = _only(MONEY_KINDS)
drill_number = _only({"percent", "degrees", "count", "quantity", "arith"})
# error category (scripts/drills/error_mix.py) -> the families that drill it, with their shares inside the category
CATEGORY_FAMILIES = {"datetime": [("clock", 0.7), ("relative", 0.15), ("from_data", 0.15)], "date": [("date", 1.0)],
                     "hh:mm": [("rule", 1.0)], "minutes": [("span", 1.0)], "money": [("money", 1.0)],
                     "number": [("number", 1.0)], "id": [("from_data", 1.0)]}


def set_mix(weights: dict) -> None:
    """Family shares from error-category weights (error-driven mix, log.md 12 Nov): replaces ``FAMILIES``."""
    by_name = {"clock": drill_clock, "relative": drill_relative, "from_data": drill_from_data, "date": drill_date,
               "rule": drill_rule, "span": drill_span, "money": drill_money, "number": drill_number}
    shares = {}
    for cat, w in weights.items():
        for fam, part in CATEGORY_FAMILIES[cat]:
            shares[fam] = shares.get(fam, 0.0) + w * part
    FAMILIES[:] = [(by_name[f], v) for f, v in shares.items() if v > 0]


FAMILIES = [(drill_span, 0.27), (drill_clock, 0.20), (drill_rule, 0.03), (drill_relative, 0.05), (drill_date, 0.18),
            (drill_from_data, 0.10), (drill_amount, 0.17)]


_HELPER = re.compile(r" (?:giúp|giùm|dùm|hộ) (?:em|anh|tôi|chị|mình|con|cô|chú|tớ|tui)\b")


def tidy_text(text: str) -> str:
    """One helper phrase per request ("xem giúp tôi ... giúp tôi nhé" -> the second one goes) and single spaces."""
    found = list(_HELPER.finditer(text))
    if len(found) > 1:
        last = found[-1]
        text = text[:last.start()] + text[last.end():]
    return " ".join(text.split())


def regional(text: str, ctx: Ctx) -> str:
    return re.sub(r"\bgiúp\b", lambda _: _pick(ctx.rng, HELPER[ctx.region]), text)


def ngrams(text: str, n: int = 8) -> set:
    w = [x.lower() for x in re.findall(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+", text)]
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def make(now: datetime, rng: random.Random, family=None, avoid: set = frozenset()) -> dict:
    """One drill; ``avoid``: 8-word sequences no drill may contain (the holdout's texts), drawn again if hit."""
    for _ in range(100):
        ctx = Ctx(now, rng)
        fn = family or _pick(rng, FAMILIES)
        d = fn(ctx)
        d["text"] = tidy_text(regional(d["text"], ctx))
        if not (ngrams(d["text"]) & avoid):
            break
    else:
        raise RuntimeError("every draw hit an avoided 8-gram")
    d["now"] = now.strftime("%Y-%m-%dT%H:%M:%S+07:00")
    d["region"] = ctx.region
    return d


def generate(contexts: list, n: int, seed: int, per_pack: tuple = (4, 6), avoid: set = frozenset()) -> list:
    """``n`` drills in packs that share one context time (sampled from ``contexts``, the training pool's current
    times). Returns packs: {"now", "drills": [...]}. The families' shares follow ``FAMILIES``; the from-data family's
    span and chain drills count towards those shares in the gate."""
    rng = random.Random(seed)
    _DIGIT_ACC.clear()
    packs, made = [], 0
    while made < n:
        now = datetime.fromisoformat(rng.choice(contexts))
        k = min(rng.randint(*per_pack), n - made)
        packs.append({"now": now.strftime("%Y-%m-%dT%H:%M:%S+07:00"), "drills": [make(now, rng, avoid=avoid) for _ in range(k)]})
        made += k
    return packs


def verify(d: dict) -> list:
    """Problems of one drill: every number, duration and clock phrase it recorded reads back to its value, and the
    phrase is in the text."""
    bad = []
    for check in d.get("checks", []):
        kind, phrase = check[0], check[1]
        if phrase not in d["text"] and phrase.capitalize() not in d["text"]:
            bad.append(f"phrase not in text: {phrase}")
        if kind == "number" and parse_number(phrase) != check[2]:
            bad.append(f"number {phrase!r} reads {parse_number(phrase)}, not {check[2]}")
        if kind == "dur" and parse_duration(phrase) != check[2]:
            bad.append(f"duration {phrase!r} reads {parse_duration(phrase)}, not {check[2]}")
        if kind == "clock" and parse_clock(phrase, check[2]) != (check[3], check[4]):
            bad.append(f"clock {phrase!r} {check[2]} reads {parse_clock(phrase, check[2])}")
    return bad


def answer_text(target: dict) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))
