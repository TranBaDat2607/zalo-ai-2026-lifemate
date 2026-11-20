#!/usr/bin/env python3
"""Synthetic conversations for the six apps held out of the fold models (experiment E2).

    python scripts/make_synth_held.py --per-scenario 7 --out src/lifemate/synth_held.jsonl

Same method as make_synth.py, which wrote the conversations for the public-only apps: requests and app data are written
here from the train tool catalog and the organisers' simulator code (the state layout in zaic_tools/apps/*.py), and
every call runs through the train copy of zaic_tools, so tool results are real. No sample that uses these apps is read:
all of them are in the validation split. The habits come from training samples of other apps: read what an ID needs,
then ask for what is still missing; pass a date when the user names a day; keep queries to the words the user said.

E2 asks whether data made this way helps apps the model has never seen. Training adds this file next to the package
synth with ``--synth package,held``.
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HELD = ["door_lock", "fitness", "learning", "library", "package_tracking", "subscriptions"]


def train_root() -> Path:
    for cand in (os.environ.get("LIFEMATE_DATA"), REPO / "train+public_test", REPO / "train+public_test" / "train+public_test"):
        if cand and (Path(cand) / "train" / "train.jsonl").is_file():
            return Path(cand)
    raise SystemExit("train data not found; set LIFEMATE_DATA")


ROOT = train_root()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ROOT / "train"))  # only the train copy of zaic_tools simulates the held-out apps
import zaic_tools.executor  # noqa: E402,F401  (imported first, so make_synth reuses this copy)
from zaic_tools.validate import catalog_tools  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import make_synth as ms  # noqa: E402
from make_synth import Ctx, Script, say_money, say_num  # noqa: E402

LOCKS = ["Cửa chính", "Cửa sau", "Cửa garage", "Cửa ban công", "Cổng trước", "Cửa phòng làm việc", "Cửa bếp"]
WORKOUTS = [("đi bộ", "walking"), ("chạy bộ", "running"), ("đạp xe", "cycling"), ("bơi", "swimming"), ("tập yoga", "yoga"),
            ("tập gym", "gym"), ("tập tạ", "gym"), ("đá bóng", "football"), ("đánh cầu lông", "badminton"),
            ("nhảy dây", "other"), ("đánh bóng bàn", "other"), ("leo núi", "other")]
COURSES = [("Tiếng Anh giao tiếp", "tiếng Anh giao tiếp"), ("Lập trình Python cơ bản", "Python"), ("Tiếng Nhật N5", "tiếng Nhật"),
           ("Guitar đệm hát", "guitar"), ("Excel văn phòng", "Excel"), ("Nhiếp ảnh căn bản", "nhiếp ảnh"), ("Tiếng Hàn sơ cấp", "tiếng Hàn"),
           ("Vẽ màu nước", "vẽ màu nước"), ("Marketing online", "marketing online"), ("Piano cho người mới", "piano"),
           ("Kỹ năng thuyết trình", "thuyết trình"), ("Toán tư duy lớp 5", "toán tư duy")]
BOOKS = [("Nhà giả kim", "Paulo Coelho"), ("Đắc nhân tâm", "Dale Carnegie"), ("Mắt biếc", "Nguyễn Nhật Ánh"),
         ("Tôi thấy hoa vàng trên cỏ xanh", "Nguyễn Nhật Ánh"), ("Cho tôi xin một vé đi tuổi thơ", "Nguyễn Nhật Ánh"),
         ("Số đỏ", "Vũ Trọng Phụng"), ("Dế mèn phiêu lưu ký", "Tô Hoài"), ("Vợ chồng A Phủ", "Tô Hoài"), ("Sapiens", "Yuval Noah Harari"),
         ("Tuổi trẻ đáng giá bao nhiêu", "Rosie Nguyễn"), ("Truyện Kiều", "Nguyễn Du"), ("Cây cam ngọt của tôi", "José Mauro de Vasconcelos"),
         ("Nghĩ giàu làm giàu", "Napoleon Hill"), ("Chí Phèo", "Nam Cao"), ("Lão Hạc", "Nam Cao"), ("Bỉ vỏ", "Nguyên Hồng")]
CARRIERS = [("Giao Hàng Nhanh", "GHN"), ("Viettel Post", "VTP"), ("J&T Express", "JT"), ("VNPost", "EB"), ("Ninja Van", "NJV"),
            ("Giao Hàng Tiết Kiệm", "GHTK"), ("SPX Express", "SPXVN")]
PARCELS = ["tai nghe", "áo khoác", "sách tiếng Anh", "nồi chiên không dầu", "giày chạy bộ", "sạc dự phòng", "bàn phím",
           "ốp lưng điện thoại", "máy xay sinh tố", "quà sinh nhật", "kem chống nắng", "đồng hồ thông minh"]
STREETS = ["18 phố Hàng Cót", "25 Lý Thường Kiệt", "102 Trần Hưng Đạo", "7 ngõ 15 Đội Cấn", "56 Nguyễn Trãi", "33 Hai Bà Trưng",
           "210 Lê Văn Sỹ", "9 Phan Chu Trinh", "41 Hùng Vương", "64 Điện Biên Phủ"]
SUBS = [("Netflix", 260000), ("Spotify Premium", 59000), ("YouTube Premium", 79000), ("iCloud 200GB", 59000), ("FPT Play", 80000),
        ("Google One", 45000), ("Apple Music", 65000), ("VieON VIP", 99000), ("ChatGPT Plus", 500000), ("Báo Tuổi Trẻ Sao", 30000),
        ("Zing MP3 VIP", 49000), ("K+", 125000)]
MUSIC = {"Spotify Premium", "Apple Music", "Zing MP3 VIP"}
_DAYS = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]


def say_past(now: datetime, d: datetime) -> str:
    delta = (now.date() - d.date()).days
    if delta == 0:
        return "hôm nay"
    if delta == 1:
        return "hôm qua"
    if delta == 2:
        return "hôm kia"
    return f"{_DAYS[d.weekday()]} vừa rồi" if delta < 7 else f"ngày {d.day} tháng {d.month}"


def say_minutes(rng: random.Random, m: int) -> str:
    if m == 30 and rng.random() < 0.5:
        return "nửa tiếng"
    if m == 60:
        return rng.choice(["một tiếng", "một giờ", "sáu mươi phút"])
    if m == 90:
        return rng.choice(["một tiếng rưỡi", "chín mươi phút"])
    return f"{say_num(rng, m)} phút"


def today_at(c: Ctx, hour: int, minute: int = 0) -> datetime:
    return c.now.replace(hour=hour, minute=minute)


def lower1(s: str) -> str:
    return s[0].lower() + s[1:]


# --- door_lock -------------------------------------------------------------------------------------------------------

def lock_state(c):
    locks = [{"id": c.ident("lock"), "name": n, "locked": c.rng.random() < 0.5, "battery": c.rng.randint(15, 100)}
             for n in c.rng.sample(LOCKS, c.rng.randint(2, 4))]
    return locks, {"locks": locks, "guest_codes": []}


def lock_set(c):
    locks, st = lock_state(c)
    lk = c.rng.choice(locks)
    want, n = not lk["locked"], lower1(lk["name"])  # the request asks for the change the state needs
    t = c.rng.choice([f"Khoá {n} {c.help()}", f"Khoá {n} lại", f"{lk['name']} khoá lại {c.help()}"]) if want else \
        c.rng.choice([f"Mở khoá {n} {c.help()}", f"Mở {n} ra", f"Mở khoá {n} cho khách vào"])
    return Script(c, {"door_lock": st}).request(c.wrap(t), [("door_lock.list_locks", {}), ("door_lock.set_lock", {"lock_id": lk["id"], "locked": want})])


def lock_check(c):
    locks, st = lock_state(c)
    lk = c.rng.choice(locks)
    t = c.rng.choice([f"{lk['name']} khoá chưa", "Kiểm tra xem các cửa khoá hết chưa", f"Xem {c.help()} {lower1(lk['name'])} đang khoá hay mở",
                      "Pin khoá cửa còn bao nhiêu"])
    return Script(c, {"door_lock": st}).request(c.wrap(t), [("door_lock.list_locks", {})])


def lock_if(c):
    locks, st = lock_state(c)
    lk = c.rng.choice(locks)
    calls = [("door_lock.list_locks", {})]
    if not lk["locked"]:
        calls.append(("door_lock.set_lock", {"lock_id": lk["id"], "locked": True}))
    t = c.rng.choice([f"Nếu {lower1(lk['name'])} chưa khoá thì khoá lại {c.help()}, khoá rồi thì thôi",
                      f"Coi {lower1(lk['name'])} còn mở không, mở thì khoá lại"])
    return Script(c, {"door_lock": st}).request(c.wrap(t), calls)


def lock_guest(c):
    locks, st = lock_state(c)
    lk = c.rng.choice(locks)
    s, d = c.later(0, 3)
    who = c.rng.choice([None, None, "chị giúp việc", "thợ sửa điện", "anh giao nước", "bạn của " + c.me])
    args = {"lock_id": lk["id"], "valid_until": c.iso(d)}
    if who:
        args["label"] = who
        t = c.rng.choice([f"Tạo mã mở {lower1(lk['name'])} cho {who}, dùng được tới {s}", f"Cấp mã tạm {lower1(lk['name'])} cho {who} hết hạn lúc {s}"])
    else:
        t = c.rng.choice([f"Tạo {c.help()} mã mở {lower1(lk['name'])} cho khách, dùng được tới {s}", f"Cho khách mã vào {lower1(lk['name'])} đến {s}"])
    return Script(c, {"door_lock": st}).request(c.wrap(t), [("door_lock.list_locks", {}), ("door_lock.grant_guest_code", args)])


def lock_guest_ask(c):
    locks, st = lock_state(c)
    lk = c.rng.choice(locks)
    t = c.rng.choice([f"Tạo {c.help()} mã mở {lower1(lk['name'])} cho khách", f"Cấp một mã tạm vào {lower1(lk['name'])}"])
    return Script(c, {"door_lock": st}).request(c.wrap(t), [("door_lock.list_locks", {}), ("sys.ask_missing_info", {"tool_name": "door_lock.grant_guest_code", "params": ["valid_until"]})])


def lock_hist(c):
    locks, st = lock_state(c)
    open_ = [lk for lk in locks if not lk["locked"]]
    if not open_:
        return None
    lk = c.rng.choice(open_)
    others = ", ".join(lower1(x["name"]) for x in open_)
    return (Script(c, {"door_lock": st})
            .turn(c.wrap(c.rng.choice(["Cửa nào đang mở vậy", "Kiểm tra mấy cái khoá cửa"])), [("door_lock.list_locks", {})], f"Dạ, đang mở: {others}.")
            .request(c.wrap(c.rng.choice([f"Khoá {lower1(lk['name'])} lại đi", f"Ừ khoá {lower1(lk['name'])} luôn"])),
                     [("door_lock.set_lock", {"lock_id": lk["id"], "locked": True})]))


# --- fitness ---------------------------------------------------------------------------------------------------------

def fit_state(c):
    days = {}
    for k in range(7):
        d = c.now - timedelta(days=k)
        steps = c.rng.randint(300, 9000) if k == 0 else c.rng.randint(1500, 14000)
        days[d.strftime("%Y-%m-%d")] = {"steps": steps, "calories": steps // 25, "distance_km": round(steps * 0.0007, 1)}
    workouts = []
    for _ in range(c.rng.randint(0, 2)):
        when = c.now - timedelta(days=c.rng.randint(1, 5))
        workouts.append({"id": c.ident("wo"), "type": c.rng.choice(WORKOUTS)[1], "minutes": c.rng.choice([20, 30, 45, 60]), "time": c.iso(when.replace(hour=6, minute=0))})
    return {"activity": days, "goal_steps": c.rng.choice([6000, 8000, 10000, 12000]), "workouts": workouts}


def fit_today(c):
    st = fit_state(c)
    if c.rng.random() < 0.25:  # no day named: no date filter
        t = c.rng.choice([f"Xem {c.help()} số bước với calo", "Đồng hồ ghi được bao nhiêu bước rồi"])
        return Script(c, {"fitness": st}).request(c.wrap(t), [("fitness.get_activity", {})])
    t = c.rng.choice([f"Hôm nay {c.me} đi được bao nhiêu bước rồi", "Hôm nay đốt được bao nhiêu calo", f"Bữa nay {c.me} đi đủ mục tiêu chưa"])
    return Script(c, {"fitness": st}).request(c.wrap(t), [("fitness.get_activity", {"date": c.now.strftime("%Y-%m-%d")})])


def fit_day(c):
    st = fit_state(c)
    d = c.now - timedelta(days=c.rng.randint(1, 6))
    day = say_past(c.now, d)
    t = c.rng.choice([f"{day.capitalize()} {c.me} đi được bao nhiêu bước", f"Xem {c.help()} {day} đi bộ được mấy cây số", f"{day.capitalize()} đốt bao nhiêu calo vậy"])
    return Script(c, {"fitness": st}).request(c.wrap(t), [("fitness.get_activity", {"date": d.strftime("%Y-%m-%d")})])


def fit_log(c):
    st = fit_state(c)
    say, kind = c.rng.choice(WORKOUTS)
    m = c.rng.choice([15, 20, 30, 40, 45, 60, 90])
    t = c.rng.choice([f"{c.me.capitalize()} vừa {say} {say_minutes(c.rng, m)}, ghi lại {c.help()}", f"Ghi buổi tập: {say} {say_minutes(c.rng, m)}",
                      f"Lưu {c.help()} là vừa {say} {say_minutes(c.rng, m)}"])
    return Script(c, {"fitness": st}).request(c.wrap(t), [("fitness.log_workout", {"type": kind, "minutes": m})])


def fit_log_time(c):
    st = fit_state(c)
    say, kind = c.rng.choice(WORKOUTS)
    m = c.rng.choice([20, 30, 45, 60, 90])
    if c.now.hour > 8 and c.rng.random() < 0.6:
        hour = c.rng.randint(5, min(c.now.hour - 2, 10))
        d, day = today_at(c, hour), "sáng nay"
    else:
        hour = c.rng.choice([5, 6, 17, 18, 19])
        d, day = (c.now - timedelta(days=1)).replace(hour=hour, minute=0), "hôm qua"
    clock = ms.say_clock(hour, 0)
    t = c.rng.choice([f"{day.capitalize()} lúc {clock} {c.me} {say} {say_minutes(c.rng, m)}, ghi vào {c.help()}",
                      f"Ghi lại buổi {say} {say_minutes(c.rng, m)} lúc {clock} {day}"])
    return Script(c, {"fitness": st}).request(c.wrap(t), [("fitness.log_workout", {"type": kind, "minutes": m, "time": c.iso(d)})])


def fit_goal(c):
    st = fit_state(c)
    steps = c.rng.choice([5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000])
    t = c.rng.choice([f"Đặt mục tiêu mỗi ngày {say_num(c.rng, steps)} bước", f"Đổi mục tiêu bước chân thành {say_num(c.rng, steps)} {c.help()}",
                      f"Mục tiêu đi bộ để {say_num(c.rng, steps)} bước một ngày"])
    return Script(c, {"fitness": st}).request(c.wrap(t), [("fitness.set_goal", {"steps": steps})])


def fit_ask(c):
    st = fit_state(c)
    if c.rng.random() < 0.5:
        say, _ = c.rng.choice(WORKOUTS)
        t = c.rng.choice([f"Ghi buổi {say} hôm nay {c.help()}", f"Lưu lại là {c.me} vừa {say} xong"])
        return Script(c, {"fitness": st}).request(c.wrap(t), [("sys.ask_missing_info", {"tool_name": "fitness.log_workout", "params": ["minutes"]})])
    t = c.rng.choice([f"Đổi mục tiêu bước chân {c.help()}", "Mục tiêu bước cao quá, hạ xuống đi"])
    return Script(c, {"fitness": st}).request(c.wrap(t), [("sys.ask_missing_info", {"tool_name": "fitness.set_goal", "params": ["steps"]})])


def fit_if_remind(c):
    st = fit_state(c)
    if c.now.hour >= 18:
        return None
    today = st["activity"][c.now.strftime("%Y-%m-%d")]
    hour = c.rng.choice([19, 20])
    calls = [("fitness.get_activity", {"date": c.now.strftime("%Y-%m-%d")})]
    if today["steps"] < st["goal_steps"]:
        calls.append(("reminders.create", {"content": "đi bộ", "time": c.iso(today_at(c, hour))}))
    t = f"Nếu hôm nay {c.me} chưa đủ mục tiêu bước thì nhắc {c.me} đi bộ lúc {ms.say_clock(hour, 0)}"
    return Script(c, {"fitness": st, "reminders": {"reminders": []}}).request(c.wrap(t), calls)


# --- learning --------------------------------------------------------------------------------------------------------

def learn_state(c):
    rows = []
    for title, spoken in c.rng.sample(COURSES, c.rng.randint(3, 6)):
        enrolled = c.rng.random() < 0.55
        rows.append(({"id": c.ident("course"), "title": title, "enrolled": enrolled, "progress_minutes": c.rng.randint(30, 900) if enrolled else 0}, spoken))
    return rows, {"courses": [r for r, _ in rows], "lessons": [], "progress": []}


def pick(rows, enrolled=None):
    good = [r for r in rows if enrolled is None or r[0]["enrolled"] == enrolled]
    return good


def learn_list(c):
    rows, st = learn_state(c)
    _, q = c.rng.choice(rows)
    t = c.rng.choice([f"{c.me.capitalize()} đang học những khoá nào", f"Có khoá học nào về {q} không", f"Khoá {q} {c.me} học được bao nhiêu rồi"])
    return Script(c, {"learning": st}).request(c.wrap(t), [("learning.list_courses", {})])


def learn_enroll(c):
    rows, st = learn_state(c)
    good = pick(rows, False)
    if not good:
        return None
    r, q = c.rng.choice(good)
    t = c.rng.choice([f"Đăng ký {c.help()} khoá {q}", f"Cho {c.me} vào học khoá {q}", f"Ghi danh khoá {q} {c.help()}"])
    return Script(c, {"learning": st}).request(c.wrap(t), [("learning.list_courses", {}), ("learning.enroll", {"course_id": r["id"]})])


def learn_schedule(c):
    rows, st = learn_state(c)
    good = pick(rows, True)
    if not good:
        return None
    r, q = c.rng.choice(good)
    s, d = c.later(0, 6)
    t = c.rng.choice([f"Đặt buổi học {q} lúc {s}", f"Xếp {c.help()} một buổi {q} vào {s}", f"Hẹn học {q} {s}"])
    return Script(c, {"learning": st}).request(c.wrap(t), [("learning.list_courses", {}), ("learning.schedule_lesson", {"course_id": r["id"], "time": c.iso(d)})])


def learn_progress(c):
    rows, st = learn_state(c)
    good = pick(rows, True)
    if not good:
        return None
    r, q = c.rng.choice(good)
    m = c.rng.choice([15, 20, 25, 30, 40, 45, 60, 90])
    t = c.rng.choice([f"{c.me.capitalize()} vừa học {q} được {say_minutes(c.rng, m)}, ghi tiến độ {c.help()}", f"Ghi là hôm nay học {q} {say_minutes(c.rng, m)}"])
    return Script(c, {"learning": st}).request(c.wrap(t), [("learning.list_courses", {}), ("learning.log_progress", {"course_id": r["id"], "minutes": m})])


def learn_ask(c):
    rows, st = learn_state(c)
    good = pick(rows, True)
    if not good:
        return None
    r, q = c.rng.choice(good)
    if c.rng.random() < 0.5:
        t = c.rng.choice([f"Đặt {c.help()} một buổi học {q}", f"Xếp lịch học {q} cho {c.me}"])
        return Script(c, {"learning": st}).request(c.wrap(t), [("learning.list_courses", {}), ("sys.ask_missing_info", {"tool_name": "learning.schedule_lesson", "params": ["time"]})])
    t = c.rng.choice([f"Ghi tiến độ khoá {q} {c.help()}", f"Cập nhật là {c.me} có học {q} rồi"])
    return Script(c, {"learning": st}).request(c.wrap(t), [("learning.list_courses", {}), ("sys.ask_missing_info", {"tool_name": "learning.log_progress", "params": ["minutes"]})])


def learn_if_enroll(c):
    rows, st = learn_state(c)
    r, q = c.rng.choice(rows)
    calls = [("learning.list_courses", {})]
    if not r["enrolled"]:
        calls.append(("learning.enroll", {"course_id": r["id"]}))
    t = c.rng.choice([f"Nếu chưa đăng ký khoá {q} thì đăng ký luôn {c.help()}", f"Coi {c.me} vào khoá {q} chưa, chưa thì ghi danh đi"])
    return Script(c, {"learning": st}).request(c.wrap(t), calls)


def learn_hist(c):
    rows, st = learn_state(c)
    good = pick(rows, False)
    if not good:
        return None
    r, q = c.rng.choice(good)
    return (Script(c, {"learning": st})
            .turn(c.wrap(f"Có khoá nào dạy {q} không"), [("learning.list_courses", {})], f"Dạ có khoá {r['title']}, {c.me} chưa đăng ký ạ.")
            .request(c.wrap(c.rng.choice(["Đăng ký khoá đó luôn", "Ừ ghi danh khoá đó đi", f"Vậy đăng ký {c.help()}"])), [("learning.enroll", {"course_id": r["id"]})]))


# --- library ---------------------------------------------------------------------------------------------------------

def lib_state(c):
    books = [{"id": c.ident("book"), "title": t, "author": a, "available": c.rng.random() < 0.65} for t, a in c.rng.sample(BOOKS, c.rng.randint(4, 8))]
    return books, {"books": books, "reservations": []}


def lib_search(c):
    books, st = lib_state(c)
    b = c.rng.choice(books)
    roll = c.rng.random()
    if roll < 0.15:  # a book the library does not have: the search is still the answer
        title = c.rng.choice([t for t, _ in BOOKS if t not in {x["title"] for x in books}])
        return Script(c, {"library": st}).request(c.wrap(f"Thư viện có cuốn {title} không"), [("library.search", {"query": title})])
    if roll < 0.55:
        t = c.rng.choice([f"Thư viện có cuốn {b['title']} không", f"Tìm {c.help()} sách {b['title']}", f"Cuốn {b['title']} còn trên kệ không"])
        return Script(c, {"library": st}).request(c.wrap(t), [("library.search", {"query": b["title"]})])
    t = c.rng.choice([f"Tìm {c.help()} sách của {b['author']}", f"Thư viện có sách nào của {b['author']} không"])
    return Script(c, {"library": st}).request(c.wrap(t), [("library.search", {"query": b["author"]})])


def lib_reserve(c):
    books, st = lib_state(c)
    b = c.rng.choice(books)
    t = c.rng.choice([f"Đặt giữ {c.help()} cuốn {b['title']}", f"Giữ cho {c.me} cuốn {b['title']} ở thư viện", f"Mượn trước cuốn {b['title']} {c.help()}"])
    return Script(c, {"library": st}).request(c.wrap(t), [("library.search", {"query": b["title"]}), ("library.reserve", {"book_id": b["id"]})])


def lib_ambiguous(c):
    books, st = lib_state(c)
    by = {}
    for b in books:
        by.setdefault(b["author"], []).append(b)
    authors = [a for a, bs in by.items() if len(bs) > 1]
    if not authors:
        return None
    a = c.rng.choice(authors)
    t = c.rng.choice([f"Giữ cho {c.me} một cuốn của {a}", f"Đặt mượn sách {a} {c.help()}"])
    return Script(c, {"library": st}).request(c.wrap(t), [("library.search", {"query": a}), ("sys.ask_missing_info", {"tool_name": "library.reserve", "params": ["book_id"]})])


def lib_if(c):
    books, st = lib_state(c)
    b = c.rng.choice(books)
    calls = [("library.search", {"query": b["title"]})]
    if b["available"]:
        calls.append(("library.reserve", {"book_id": b["id"]}))
    t = c.rng.choice([f"Nếu cuốn {b['title']} đang có sẵn thì giữ {c.help()}, không thì thôi", f"Coi {b['title']} còn không, còn thì đặt giữ luôn"])
    return Script(c, {"library": st}).request(c.wrap(t), calls)


def lib_hist(c):
    books, st = lib_state(c)
    b = c.rng.choice(books)
    state = "đang có sẵn" if b["available"] else "đang có người mượn"
    return (Script(c, {"library": st})
            .turn(c.wrap(f"Tìm sách {b['title']}"), [("library.search", {"query": b["title"]})], f"Dạ có cuốn {b['title']} của {b['author']}, {state} ạ.")
            .request(c.wrap(c.rng.choice(["Ừ đặt giữ cuốn đó đi", "Giữ cuốn đó cho " + c.me, "Vậy đặt mượn luôn"])), [("library.reserve", {"book_id": b["id"]})]))


# --- package_tracking ------------------------------------------------------------------------------------------------

def pkg_state(c):
    rows = []
    for name in c.rng.sample(PARCELS, c.rng.randint(2, 4)):
        carrier, prefix = c.rng.choice(CARRIERS)
        status = c.rng.choice(["in_transit", "in_transit", "out_for_delivery", "delivered"])
        eta = c.now + timedelta(days=c.rng.randint(0, 3)) if status != "delivered" else c.now - timedelta(days=c.rng.randint(1, 3))
        rows.append({"id": c.ident("shp"), "tracking_id": prefix + "".join(str(c.rng.randint(0, 9)) for _ in range(9)), "carrier": carrier,
                     "name": name, "status": status, "eta": eta.strftime("%Y-%m-%d")})
    return rows, {"shipments": rows, "pickups": []}


def pkg_all(c):
    _, st = pkg_state(c)
    t = c.rng.choice([f"Mấy đơn hàng của {c.me} tới đâu rồi", "Kiểm tra các kiện hàng đang giao", f"Hàng của {c.me} hôm nay có tới không"])
    return Script(c, {"package_tracking": st}).request(c.wrap(t), [("package_tracking.track", {})])


def pkg_one(c):
    rows, st = pkg_state(c)
    p = c.rng.choice(rows)
    if c.rng.random() < 0.5:  # the name is not a parameter: list everything
        t = c.rng.choice([f"Đơn {p['name']} giao tới đâu rồi", f"Kiện {p['name']} bao giờ tới"])
        return Script(c, {"package_tracking": st}).request(c.wrap(t), [("package_tracking.track", {})])
    t = c.rng.choice([f"Kiểm tra vận đơn {p['tracking_id']}", f"Tra {c.help()} mã vận đơn {p['tracking_id']}"])
    return Script(c, {"package_tracking": st}).request(c.wrap(t), [("package_tracking.track", {"tracking_id": p["tracking_id"]})])


def pkg_reschedule(c):
    rows, st = pkg_state(c)
    live = [p for p in rows if p["status"] != "delivered"]
    if not live:
        return None
    p = c.rng.choice(live)
    s, d = c.later(0, 3)
    t = c.rng.choice([f"Hẹn giao lại đơn {p['name']} vào {s}", f"Đơn {p['name']} giao {s} {c.help()}", f"Dời giờ giao kiện {p['name']} sang {s}"])
    return Script(c, {"package_tracking": st}).request(c.wrap(t), [("package_tracking.track", {}), ("package_tracking.reschedule_delivery", {"tracking_id": p["tracking_id"], "time": c.iso(d)})])


def pkg_pickup(c):
    _, st = pkg_state(c)
    day, d = c.day(0, 5)
    where, addr = c.rng.choice([("ở nhà", c.home), ("ở công ty", c.work)] + [(f"ở {x}", x) for x in c.rng.sample(STREETS, 2)])
    t = c.rng.choice([f"Hẹn bưu tá tới lấy hàng {where} {day}", f"Đặt lịch lấy hàng gửi đi {where} {day} {c.help()}"])
    return Script(c, {"package_tracking": st}).request(c.wrap(t), [("package_tracking.schedule_pickup", {"address": addr, "date": d.strftime("%Y-%m-%d")})])


def pkg_pickup_ask(c):
    _, st = pkg_state(c)
    day, _ = c.day(0, 5)
    addr = c.rng.choice(STREETS)
    t, params = c.rng.choice([(f"Hẹn bưu tá tới lấy hàng {day}", ["address"]), (f"Hẹn lấy hàng gửi đi ở {addr}", ["date"]),
                              (f"Gọi bưu tá tới lấy hàng {c.help()}", ["address", "date"])])
    return Script(c, {"package_tracking": st}).request(c.wrap(t), [("sys.ask_missing_info", {"tool_name": "package_tracking.schedule_pickup", "params": params})])


def pkg_if(c):
    rows, st = pkg_state(c)
    p = c.rng.choice(rows)
    s, d = c.later(0, 2)
    calls = [("package_tracking.track", {})]
    if p["status"] != "delivered":
        calls.append(("package_tracking.reschedule_delivery", {"tracking_id": p["tracking_id"], "time": c.iso(d)}))
    t = f"Nếu đơn {p['name']} chưa giao thì hẹn giao lại {s}, giao rồi thì thôi"
    return Script(c, {"package_tracking": st}).request(c.wrap(t), calls)


# --- subscriptions ---------------------------------------------------------------------------------------------------

def sub_state(c):
    rows = [{"id": c.ident("sub"), "name": n, "price": p, "next_billing": (c.now + timedelta(days=c.rng.randint(1, 30))).strftime("%Y-%m-%d"),
             "status": "active" if c.rng.random() < 0.85 else "cancelled"} for n, p in c.rng.sample(SUBS, c.rng.randint(3, 6))]
    return rows, {"subscriptions": rows}


def sub_list(c):
    rows, st = sub_state(c)
    s = c.rng.choice(rows)
    t = c.rng.choice([f"{c.me.capitalize()} đang đăng ký những dịch vụ nào", "Tháng này gói nào sắp gia hạn", f"Gói {s['name']} bao nhiêu tiền một tháng",
                      f"Mỗi tháng {c.me} trả bao nhiêu cho mấy gói đăng ký"])
    return Script(c, {"subscriptions": st}).request(c.wrap(t), [("subscriptions.list", {})])


def sub_cancel(c):
    rows, st = sub_state(c)
    live = [s for s in rows if s["status"] == "active" and s["name"] not in MUSIC]
    if not live:
        return None
    s = c.rng.choice(live)
    t = c.rng.choice([f"Huỷ {c.help()} gói {s['name']}", f"Thôi không dùng {s['name']} nữa, huỷ đi", f"Ngừng gia hạn {s['name']} {c.help()}"])
    return Script(c, {"subscriptions": st}).request(c.wrap(t), [("subscriptions.list", {}), ("subscriptions.cancel", {"subscription_id": s["id"]})])


def sub_if_price(c):
    rows, st = sub_state(c)
    s = c.rng.choice([x for x in rows if x["status"] == "active"] or rows)
    limit = c.rng.choice([50000, 60000, 80000, 100000, 200000])
    calls = [("subscriptions.list", {})]
    if s["status"] == "active" and s["price"] > limit:
        calls.append(("subscriptions.cancel", {"subscription_id": s["id"]}))
    t = f"Nếu gói {s['name']} trên {say_money(c.rng, limit)} một tháng thì huỷ đi, không thì giữ"
    return Script(c, {"subscriptions": st}).request(c.wrap(t), calls)


def sub_music(c):
    rows, st = sub_state(c)
    music = [s for s in rows if s["name"] in MUSIC and s["status"] == "active"]
    if not music:
        return None
    t = c.rng.choice(["Huỷ gói nghe nhạc đi", f"Huỷ {c.help()} cái app nhạc trả tiền"])
    last = ("subscriptions.cancel", {"subscription_id": music[0]["id"]}) if len(music) == 1 else \
        ("sys.ask_missing_info", {"tool_name": "subscriptions.cancel", "params": ["subscription_id"]})
    return Script(c, {"subscriptions": st}).request(c.wrap(t), [("subscriptions.list", {}), last])


def sub_hist(c):
    rows, st = sub_state(c)
    live = [s for s in rows if s["status"] == "active"]
    if not live:
        return None
    s = c.rng.choice(live)
    names = ", ".join(x["name"] for x in live)
    return (Script(c, {"subscriptions": st})
            .turn(c.wrap(f"{c.me.capitalize()} đang có những gói nào"), [("subscriptions.list", {})], f"Dạ đang có: {names}.")
            .request(c.wrap(c.rng.choice([f"Huỷ cái {s['name']} đi", f"Bỏ {s['name']} giúp {c.me}"])), [("subscriptions.cancel", {"subscription_id": s["id"]})]))


# --- combinations with apps the model knows ---------------------------------------------------------------------------

def combo_lock_alarm(c):
    locks, st = lock_state(c)
    open_ = [lk for lk in locks if not lk["locked"]]
    if not open_:
        return None
    lk = c.rng.choice(open_)
    s, d = c.later(1, 1)
    t = f"Khoá {lower1(lk['name'])} rồi đặt báo thức {s} {c.help()}"
    return Script(c, {"door_lock": st, "alarm": {"alarms": []}}).request(
        c.wrap(t), [("door_lock.list_locks", {}), ("door_lock.set_lock", {"lock_id": lk["id"], "locked": True}), ("alarm.create", {"time": c.iso(d)})])


def combo_lib_remind(c):
    books, st = lib_state(c)
    b = c.rng.choice(books)
    s, d = c.later(1, 3)
    t = f"Đặt giữ cuốn {b['title']} rồi nhắc {c.me} đi lấy sách lúc {s}"
    return Script(c, {"library": st, "reminders": {"reminders": []}}).request(
        c.wrap(t), [("library.search", {"query": b["title"]}), ("library.reserve", {"book_id": b["id"]}), ("reminders.create", {"content": "đi lấy sách", "time": c.iso(d)})])


def combo_learn_calendar(c):
    rows, st = learn_state(c)
    good = pick(rows, True)
    if not good:
        return None
    r, q = c.rng.choice(good)
    s, d = c.later(1, 6)
    t = f"Đặt buổi học {q} lúc {s} rồi ghi vào lịch {c.help()}"
    return Script(c, {"learning": st, "calendar": {"events": []}}).request(
        c.wrap(t), [("learning.list_courses", {}), ("learning.schedule_lesson", {"course_id": r["id"], "time": c.iso(d)}),
                    ("calendar.create_event", {"title": f"Học {q}", "start": c.iso(d)})])


def combo_sub_sms(c):
    rows, st = sub_state(c)
    live = [s for s in rows if s["status"] == "active" and s["name"] not in MUSIC]
    if not live:
        return None
    s = c.rng.choice(live)
    contacts = ms.contact_rows(c)
    who = contacts[0]
    t = f"Huỷ gói {s['name']} rồi nhắn {who['name']} là {c.me} huỷ {s['name']} rồi"
    return Script(c, {"subscriptions": st, "contacts": {"contacts": contacts}, "sms": {"messages": []}}).request(
        c.wrap(t), [("subscriptions.list", {}), ("subscriptions.cancel", {"subscription_id": s["id"]}), ("contacts.search", {"query": who["name"]}),
                    ("sms.send", {"contact_id": who["id"], "content": f"{c.me} huỷ {s['name']} rồi"})])


def combo_fit_note(c):
    st = fit_state(c)
    say, kind = c.rng.choice(WORKOUTS)
    m = c.rng.choice([30, 45, 60])
    s, d = c.later(1, 1)
    t = f"Ghi {c.help()} buổi {say} {say_minutes(c.rng, m)} vừa xong, rồi nhắc {c.me} {say} tiếp lúc {s}"
    return Script(c, {"fitness": st, "reminders": {"reminders": []}}).request(
        c.wrap(t), [("fitness.log_workout", {"type": kind, "minutes": m}), ("reminders.create", {"content": say, "time": c.iso(d)})])


def not_installed(c):
    """The app that could do this is missing, so the right answer is no call at all."""
    t, used = c.rng.choice([(f"Khoá cửa chính {c.help()}", "door_lock"), ("Hôm nay đi được bao nhiêu bước", "fitness"),
                            ("Đăng ký khoá tiếng Anh giao tiếp", "learning"), ("Thư viện có cuốn Mắt biếc không", "library"),
                            ("Đơn hàng của tôi tới đâu rồi", "package_tracking"), ("Huỷ gói Netflix", "subscriptions")])
    s = Script(c, {}).request(c.wrap(t), [])
    s.exclude = used
    return s


SCENARIOS = [lock_set, lock_check, lock_if, lock_guest, lock_guest_ask, lock_hist,
             fit_today, fit_day, fit_log, fit_log_time, fit_goal, fit_ask, fit_if_remind,
             learn_list, learn_enroll, learn_schedule, learn_progress, learn_ask, learn_if_enroll, learn_hist,
             lib_search, lib_reserve, lib_ambiguous, lib_if, lib_hist,
             pkg_all, pkg_one, pkg_reschedule, pkg_pickup, pkg_pickup_ask, pkg_if,
             sub_list, sub_cancel, sub_if_price, sub_music, sub_hist,
             combo_lock_alarm, combo_lib_remind, combo_learn_calendar, combo_sub_sms, combo_fit_note, not_installed]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=7)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default=str(REPO / "src" / "lifemate" / "synth_held.jsonl"))
    args = ap.parse_args()
    catalog = catalog_tools(json.loads((ROOT / "train" / "train_tools_catalog.json").read_text(encoding="utf-8")))
    # Distractors come from the train catalog: the package list also names public-only apps, which have no tools here.
    ms.DISTRACTORS = sorted({n.split(".", 1)[0] for n in catalog} - {"sys"})
    rng = random.Random(args.seed)
    rows, failures = [], {}
    for scenario in SCENARIOS:
        made, tries = 0, 0
        while made < args.per_scenario and tries < args.per_scenario * 10:
            tries += 1
            try:
                script = scenario(Ctx(rng))
                if script is None:
                    continue
                rows.append(ms.build_sample(script, f"synth_held_{len(rows):05d}", catalog))
                made += 1
            except (ValueError, TypeError, IndexError, RuntimeError) as e:
                failures[scenario.__name__] = f"{type(e).__name__}: {e}"
        if made < args.per_scenario:
            print(f"{scenario.__name__}: only {made}", flush=True)
    rng.shuffle(rows)
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} samples from {len(SCENARIOS)} scenarios -> {args.out}")
    for k, v in failures.items():
        print("failed sometimes:", k, v[:200])


if __name__ == "__main__":
    main()
