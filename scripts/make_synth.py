#!/usr/bin/env python3
"""Generate synthetic training conversations for the apps that train does not cover.

    python scripts/make_synth.py --per-scenario 12 --out src/lifemate/synth_samples.jsonl

Requests and app data are written here from scratch and run through the organisers' zaic_tools code, so the gold tool
results are real. Nothing under <split>_environment/ or any public test sample is read. Calls follow the habits seen in
train: read what an ID needs, keep search queries short, pass no optional filter the user did not ask for.
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
PUBLIC = Path(os.environ.get("LIFEMATE_DATA", REPO / "train+public_test")) / "public_test"  # the data root, nested or not
sys.path.insert(0, str(PUBLIC))

from lifemate.asr_eval import spell_number  # noqa: E402
from zaic_tools.executor import Simulator  # noqa: E402
from zaic_tools.validate import catalog_tools  # noqa: E402

TZ = "+07:00"
PROFILES = [
    ("Anh Nam", "25 Láng Hạ, Đống Đa, Hà Nội", "Tòa nhà Kim Cương, Nam Từ Liêm, Hà Nội", "Hà Nội", 21.0285, 105.8542),
    ("Chị Hoa", "12 Nguyễn Huệ, Quận 1, TP.HCM", "Khu Công nghệ cao, TP. Thủ Đức, TP.HCM", "TP.HCM", 10.7769, 106.7009),
    ("Em Linh", "88 Nguyễn Văn Linh, Hải Châu, Đà Nẵng", "Công viên phần mềm, Hải Châu, Đà Nẵng", "Đà Nẵng", 16.0544, 108.2022),
    ("Anh Khoa", "7 Lê Lợi, Phú Nhuận, Huế", "Khu đô thị An Cựu, Huế", "Huế", 16.4637, 107.5909),
    ("Bác Tư", "45 Ba Tháng Hai, Ninh Kiều, Cần Thơ", "Chợ Cái Khế, Ninh Kiều, Cần Thơ", "Cần Thơ", 10.0452, 105.7469),
    ("Cô Thảo", "19 Trần Phú, Ba Đình, Hà Nội", "Trường THCS Giảng Võ, Ba Đình, Hà Nội", "Hà Nội", 21.0285, 105.8542),
]
PRON = {"Anh": ["anh", "tôi"], "Chị": ["chị", "tôi", "mình"], "Em": ["em", "mình"], "Bác": ["bác", "tôi"], "Cô": ["cô", "tôi"]}
LEADS = ["", "", "", "", "Ơi, ", "Em ơi, ", "Trợ lý ơi, ", "Ờ, ", "À, ", "Này, ", "Bạn ơi, ", "Ờ à, "]
TAILS = ["", " nhé", " nha", " nghe", " với", " nhé em", " nha em", " đi", " hen"]
DISTRACTORS = ["weather", "calendar", "contacts", "sms", "smart_home", "tv", "gold_price", "banking", "ewallet", "alarm", "reminders",
               "notes", "tasks", "music", "phone", "email", "chat_ott", "maps", "fuel_price", "food_delivery", "ride_hailing", "flights",
               "shopping_list", "bill_payment", "grocery", "recipes", "parking", "public_transit", "pantry", "train_bus_tickets",
               "ecommerce", "event_tickets", "stocks", "translator", "habit_tracker"]

GROCERY = [("Gạo ST25 túi 5kg", "gạo ST25", "túi", 185000), ("Nước mắm Nam Ngư chai 500ml", "nước mắm", "chai", 45000),
           ("Trứng gà ta hộp 10 quả", "trứng gà", "hộp", 42000), ("Thịt ba chỉ khay 500g", "thịt ba chỉ", "khay", 89000),
           ("Cá hồi phi lê khay 300g", "cá hồi", "khay", 159000), ("Sữa tươi hộp 1 lít", "sữa tươi", "hộp", 36000),
           ("Rau muống bó 500g", "rau muống", "bó", 12000), ("Cà chua Đà Lạt 1kg", "cà chua", "ký", 32000),
           ("Hành tây 1kg", "hành tây", "ký", 25000), ("Dầu ăn chai 1 lít", "dầu ăn", "chai", 62000),
           ("Mì tôm thùng 30 gói", "mì tôm", "thùng", 115000), ("Nước suối chai 500ml", "nước suối", "chai", 6000),
           ("Đùi gà góc tư 1kg", "đùi gà", "khay", 89000), ("Sữa chua lốc 4 hộp", "sữa chua", "lốc", 28000),
           ("Bánh mì sandwich gói", "bánh mì", "gói", 25000), ("Táo Envy 1kg", "táo", "ký", 149000),
           ("Đậu hũ non hộp", "đậu hũ", "hộp", 15000), ("Nấm kim châm gói 200g", "nấm kim châm", "gói", 18000),
           ("Tôm thẻ khay 500g", "tôm thẻ", "khay", 135000), ("Cà phê phin gói 500g", "cà phê", "gói", 98000)]
ECOM = [("Tai nghe Bluetooth Sony WH-CH520", "tai nghe", 1290000), ("Bàn phím cơ AKKO 3068B", "bàn phím", 1290000),
        ("Chuột không dây Logitech M331", "chuột không dây", 390000), ("Nồi chiên không dầu Philips 4.1L", "nồi chiên", 2490000),
        ("Máy xay sinh tố Sunhouse", "máy xay sinh tố", 690000), ("Cốc giữ nhiệt Elmich 500ml", "cốc giữ nhiệt", 219000),
        ("Áo khoác gió CoolWear", "áo khoác", 499000), ("Sạc dự phòng Anker 10000mAh", "sạc dự phòng", 590000),
        ("Đèn đọc sách Xiaomi", "đèn đọc sách", 329000), ("Ba lô laptop Tomtoc", "ba lô laptop", 890000),
        ("Giày chạy bộ Nike Pegasus", "giày chạy bộ", 2890000), ("Ốp lưng iPhone 15", "ốp lưng", 150000),
        ("Bình đun siêu tốc Lock&Lock", "bình đun", 450000), ("Quạt mini cầm tay Baseus", "quạt mini", 280000)]
RECIPES = [("Canh chua cá lóc", 4, ["cá lóc", "dứa", "cà chua", "giá đỗ"]), ("Thịt kho trứng", 4, ["thịt ba chỉ", "trứng vịt", "nước dừa"]),
           ("Gà xào sả ớt", 3, ["đùi gà", "sả", "ớt"]), ("Phở bò", 4, ["bánh phở", "thịt bò", "hành tây", "gừng"]),
           ("Bún chả", 4, ["bún", "thịt ba chỉ", "nước mắm", "đu đủ"]), ("Trứng chiên cà chua", 2, ["trứng gà", "cà chua", "hành lá"]),
           ("Cơm rang dưa bò", 2, ["cơm nguội", "thịt bò", "dưa cải"]), ("Lẩu nấm chay", 4, ["nấm kim châm", "nấm đùi gà", "đậu hũ"]),
           ("Bánh pancake chuối", 2, ["bột mì", "chuối", "trứng gà", "sữa tươi"]), ("Cháo nấm", 4, ["gạo", "nấm rơm", "hành lá"])]
PANTRY = [("Trứng gà", "quả"), ("Sữa tươi", "hộp"), ("Gạo ST25", "kg"), ("Mì gói", "gói"), ("Bột mì", "gói"), ("Đường", "kg"),
          ("Cà phê", "gói"), ("Nước mắm", "chai"), ("Dầu ăn", "chai"), ("Yến mạch", "túi"), ("Đậu xanh", "gói"), ("Cà chua", "quả")]
HABITS = ["Chạy bộ buổi sáng", "Đọc sách 30 phút", "Thiền 10 phút", "Uống đủ 2 lít nước", "Ngủ trước 11 giờ", "Học tiếng Anh 20 phút", "Tập yoga"]
LOTS = {"Hà Nội": [("Bãi đỗ Cung Triển lãm", "Phạm Hùng, Nam Từ Liêm", "Mỹ Đình"), ("Bãi xe Ga Hà Nội", "120 Lê Duẩn, Hoàn Kiếm", "Hoàn Kiếm"), ("Bãi đỗ Hồ Gươm", "2 Lê Thái Tổ, Hoàn Kiếm", "Hồ Gươm")],
        "TP.HCM": [("Bãi đỗ Nam Đô", "2 Lê Lợi, Quận 1", "Lê Lợi"), ("Bãi xe Bến Thành", "Phan Bội Châu, Quận 1", "Bến Thành"), ("Bãi đỗ Thảo Điền", "Xuân Thủy, Thủ Đức", "Thảo Điền")],
        "Đà Nẵng": [("Bãi đỗ Chợ Hàn", "119 Trần Phú, Hải Châu", "chợ Hàn"), ("Bãi xe Cầu Rồng", "Nguyễn Văn Linh, Hải Châu", "Cầu Rồng")],
        "Huế": [("Bãi xe Chợ Đông Ba", "Trần Hưng Đạo, Huế", "chợ Đông Ba"), ("Bãi đỗ Đại Nội", "Đoàn Thị Điểm, Huế", "Đại Nội")],
        "Cần Thơ": [("Bãi xe Bến Ninh Kiều", "Hai Bà Trưng, Ninh Kiều", "bến Ninh Kiều"), ("Bãi đỗ Cái Răng", "Võ Tánh, Cái Răng", "Cái Răng")]}
STOPS = {"Hà Nội": ["Bến xe Nam Thăng Long", "Cầu Giấy", "Kim Mã", "Bờ Hồ", "Bệnh viện Bạch Mai", "Giáp Bát"],
         "TP.HCM": ["Bến Thành", "Suối Tiên", "Thảo Điền", "Bến xe Miền Đông", "Chợ Lớn", "Đầm Sen"],
         "Đà Nẵng": ["Công viên 29/3", "Cầu Rồng", "Chợ Hàn", "Sân bay Đà Nẵng", "Bến xe Trung tâm"],
         "Huế": ["Nhà hát Sông Hương", "Chợ Đông Ba", "Ga Huế", "Đại Nội"], "Cần Thơ": ["Bến Ninh Kiều", "Chợ Cái Khế", "Cầu Cần Thơ", "Đại học Cần Thơ"]}
ROUTES = ["Hà Nội", "TP.HCM", "Đà Nẵng", "Huế", "Vinh", "Nha Trang", "Hải Phòng", "Đà Lạt", "Lào Cai", "Quy Nhơn", "Cần Thơ"]
OPERATORS = {"train": ["SE1", "SE3", "SE5", "SE7", "SE9", "TN2"], "bus": ["Hoàng Long", "Thành Bưởi", "Phương Trang", "Hoàng Vân", "Kumho"]}
EVENTS = [("Đêm nhạc Trịnh Công Sơn", "ca nhạc", "Nhà hát Lớn"), ("Giải bóng rổ thành phố", "thể thao", "Cung thể thao Quần Ngựa"),
          ("Triển lãm tranh Đông Dương", "triển lãm", "Bảo tàng Mỹ thuật"), ("Lễ hội ẩm thực mùa thu", "lễ hội", "Công viên Thống Nhất"),
          ("Vở kịch Lôi Vũ", "kịch", "Nhà hát Tuổi Trẻ"), ("Phim Mùa Xuân Sớm", "phim", "Rạp CineViệt"),
          ("Hội chợ sách mùa thu", "hội chợ", "Cung Văn hóa Hữu Nghị"), ("Workshop nhiếp ảnh đường phố", "workshop", "Không gian Sáng Tạo")]
STOCKS = [("MLG", "Thép Minh Long"), ("BKV", "Ngân hàng Bắc Việt"), ("VTX", "Viettex"), ("HSP", "Hải Sản Phú Quý"), ("TDS", "Thế Giới Đồ Số"),
          ("NKL", "Nhựa Kim Long"), ("DPH", "Dược Phúc Hưng"), ("GMT", "Vận Tải Gemtrans"), ("SPK", "Sao Phương Khoa"), ("ANB", "An Bình Land")]
SENTENCES = ["Tôi bị dị ứng hải sản", "Cho tôi xem thực đơn", "Nhà vệ sinh ở đâu", "Tôi sẽ đến muộn mười lăm phút", "Hóa đơn này bao nhiêu tiền",
             "Xin lỗi, tôi không hiểu", "Cho tôi một ly cà phê sữa đá", "Chúng tôi đặt bàn cho bốn người", "Món này có cay không", "Làm ơn gọi taxi giúp tôi"]
LANGS = [("tiếng Anh", "en"), ("tiếng Hàn", "ko"), ("tiếng Nhật", "ja"), ("tiếng Trung", "zh"), ("tiếng Pháp", "fr")]
CONTACTS = ["Mẹ", "Bố", "Lan", "Minh", "Tuấn", "Hương", "Sếp Dũng", "Bác Ba", "Cô giáo Mai", "Anh Hùng", "Thảo", "Vợ Hằng"]


class Ctx:
    def __init__(self, rng: random.Random):
        self.rng = rng
        p = rng.choice(PROFILES)
        self.name, self.home, self.work, self.city, self.lat, self.lng = p
        self.me = rng.choice(PRON[self.name.split()[0]])
        month = rng.choice([9, 10, 11, 11, 12])
        self.now = datetime(2026, month, rng.randint(1, 28), rng.randint(7, 21), rng.choice([0, 0, 10, 15, 30, 40, 45]))
        self.seq = {}

    def ident(self, prefix: str) -> str:
        """Both ID styles of the data: short sequential and long random."""
        self.seq[prefix] = self.seq.get(prefix, 0) + 1
        if self.rng.random() < 0.5:
            return f"{prefix}_{self.seq[prefix]:02d}"
        return f"{prefix}_{self.rng.randint(100, 9999)}"

    def wrap(self, body: str) -> str:
        lead = self.rng.choice(LEADS)
        body = " ".join(body.split())
        if lead:
            body = body[0].lower() + body[1:]
        return lead + body + self.rng.choice(TAILS) + "."

    def help(self) -> str:
        return self.rng.choice([f"giúp {self.me}", f"giùm {self.me}", f"dùm {self.me}", f"hộ {self.me}", f"cho {self.me}", ""]).strip()

    def iso(self, d: datetime) -> str:
        return d.strftime("%Y-%m-%dT%H:%M:00") + TZ

    def later(self, min_days: int = 0, max_days: int = 5) -> tuple:
        """(spoken, datetime) of a time after now."""
        for _ in range(50):
            days = self.rng.randint(min_days, max_days)
            hour, minute = self.rng.randint(6, 21), self.rng.choice([0, 0, 0, 30, 15])
            d = (self.now + timedelta(days=days)).replace(hour=hour, minute=minute)
            if d > self.now + timedelta(minutes=30):
                return f"{say_clock(hour, minute)} {say_day(self.now, d)}", d
        raise RuntimeError("no slot")

    def day(self, min_days: int = 1, max_days: int = 7) -> tuple:
        d = self.now + timedelta(days=self.rng.randint(min_days, max_days))
        return say_day(self.now, d), d


def say_num(rng: random.Random, n: int) -> str:
    return " ".join(spell_number(str(n))) if rng.random() < 0.7 else str(n)


def say_money(rng: random.Random, n: int) -> str:
    if n >= 1000000:
        tr, rest = n // 1000000, n % 1000000
        text = " ".join(spell_number(str(tr))) + " triệu"
        if rest == 500000:
            return text + " rưỡi"
        if rest:
            text += " " + " ".join(spell_number(str(rest // 1000))) + " nghìn"
        return text
    return " ".join(spell_number(str(n // 1000))) + rng.choice([" nghìn", " nghìn", " ngàn"])


def say_clock(hour: int, minute: int) -> str:
    part = "sáng" if hour < 11 else ("trưa" if hour < 13 else ("chiều" if hour < 18 else "tối"))
    h12 = hour if hour <= 12 else hour - 12
    words = " ".join(spell_number(str(h12)))
    if minute == 30:
        return f"{words} rưỡi {part}"
    if minute:
        return f"{words} giờ {' '.join(spell_number(str(minute)))} {part}"
    return f"{words} giờ {part}"


_DAYS = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]


def say_day(now: datetime, d: datetime) -> str:
    delta = (d.date() - now.date()).days
    if delta == 0:
        return "hôm nay"
    if delta == 1:
        return "mai"
    if delta == 2:
        return "ngày kia"
    return f"{_DAYS[d.weekday()]} ngày {d.day} tháng {d.month}"


class Script:
    """One conversation: earlier turns (already applied to the state) and the scored request."""

    def __init__(self, ctx: Ctx, states: dict):
        self.ctx, self.states = ctx, states
        self.history, self.final_text, self.final_calls = [], "", []

    def turn(self, text: str, calls: list, reply: str):
        self.history.append((text, calls, reply))
        return self

    def request(self, text: str, calls: list):
        self.final_text, self.final_calls = text, calls
        return self


def stock_products(c: Ctx, base: list, prefix: str, k=(4, 8), extra=None) -> list:
    rows = []
    for name, q, *rest in c.rng.sample(base, c.rng.randint(*k)):
        row = {"id": c.ident(prefix), "name": name, "price": rest[-1], "in_stock": c.rng.random() > 0.12}
        row.update(extra(q, rest) if extra else {})
        rows.append((q, row))
    return rows


def groc_state(c):
    rows = stock_products(c, GROCERY, "gro", extra=lambda q, r: {"unit": r[0], "category": "thực phẩm"})
    return rows, {"products": [r for _, r in rows], "cart": [], "orders": []}


def ecom_state(c):
    rows = stock_products(c, ECOM, "prod", extra=lambda q, r: {"seller": "Shop chính hãng", "category": "gia dụng", "rating": round(c.rng.uniform(4.1, 4.9), 1)})
    orders = []
    for i in range(c.rng.randint(1, 3)):
        _, p = c.rng.choice(rows)
        orders.append({"id": c.ident("ord"), "items": [{"product_id": p["id"], "name": p["name"], "price": p["price"], "quantity": 1}], "total": p["price"],
                       "status": c.rng.choice(["placed", "shipping", "delivered"]), "placed_at": c.iso(c.now - timedelta(days=c.rng.randint(1, 9)))})
    return rows, {"products": [r for _, r in rows], "cart": [], "orders": orders, "returns": []}


def need_stock(rows, c):
    in_stock = [r for r in rows if r[1]["in_stock"]]
    return c.rng.choice(in_stock)


# --- scenarios: each returns a Script, or None when the random state does not suit it ---

def grocery_search(c):
    rows, st = groc_state(c)
    q, _ = c.rng.choice(rows)
    t = c.rng.choice([f"Tìm {c.help()} {q} trên đi chợ online xem còn không", f"Xem {c.help()} {q} bên siêu thị giá bao nhiêu", f"Đi chợ online có {q} không"])
    return Script(c, {"grocery": st}).request(c.wrap(t), [("grocery.search", {"query": q})])


def grocery_add(c):
    rows, st = groc_state(c)
    q, p = need_stock(rows, c)
    n = c.rng.randint(1, 5)
    t = c.rng.choice([f"Thêm {say_num(c.rng, n)} {p['unit']} {q} vào giỏ đi chợ {c.help()}", f"Cho {say_num(c.rng, n)} {p['unit']} {q} vô giỏ hàng siêu thị", f"Bỏ vô giỏ {c.help()} {say_num(c.rng, n)} {p['unit']} {q}"])
    return Script(c, {"grocery": st}).request(c.wrap(t), [("grocery.search", {"query": q}), ("grocery.add_to_cart", {"product_id": p["id"], "quantity": n})])


def grocery_buy(c):
    rows, st = groc_state(c)
    (q1, p1), (q2, p2) = c.rng.sample([r for r in rows if r[1]["in_stock"]], 2) if sum(r[1]["in_stock"] for r in rows) > 1 else (None, None)
    n1, n2 = c.rng.randint(1, 3), c.rng.randint(1, 3)
    calls = [("grocery.search", {"query": q1}), ("grocery.add_to_cart", {"product_id": p1["id"], "quantity": n1}),
             ("grocery.search", {"query": q2}), ("grocery.add_to_cart", {"product_id": p2["id"], "quantity": n2})]
    s, d = c.later(0, 2)
    style = c.rng.randint(0, 2)
    if style == 0:
        t, args = f"Mua {c.help()} {say_num(c.rng, n1)} {p1['unit']} {q1} với {say_num(c.rng, n2)} {p2['unit']} {q2} rồi đặt luôn", {}
    elif style == 1:
        t, args = f"Đi chợ {c.help()} {say_num(c.rng, n1)} {p1['unit']} {q1}, {say_num(c.rng, n2)} {p2['unit']} {q2}, giao về nhà lúc {s}", {"address": c.home, "delivery_time": c.iso(d)}
    else:
        t, args = f"Đặt {say_num(c.rng, n1)} {p1['unit']} {q1} và {say_num(c.rng, n2)} {p2['unit']} {q2} giao tới công ty", {"address": c.work}
    return Script(c, {"grocery": st}).request(c.wrap(t), calls + [("grocery.checkout", args)])


def grocery_after_search(c):
    rows, st = groc_state(c)
    q, p = need_stock(rows, c)
    n = c.rng.randint(1, 4)
    text = c.rng.choice([f"Ừ lấy {say_num(c.rng, n)} {p['unit']} đó đi", f"Thế cho {say_num(c.rng, n)} {p['unit']} vô giỏ rồi thanh toán luôn"])
    calls = [("grocery.add_to_cart", {"product_id": p["id"], "quantity": n})]
    if "thanh toán" in text:
        calls.append(("grocery.checkout", {}))
    return (Script(c, {"grocery": st})
            .turn(c.wrap(f"Siêu thị có {q} không"), [("grocery.search", {"query": q})], f"Dạ có {p['name']}, giá {p['price']} đồng ạ.")
            .request(c.wrap(text), calls))


def grocery_if_stock(c):
    rows, st = groc_state(c)
    q, p = c.rng.choice(rows)
    n = c.rng.randint(1, 3)
    calls = [("grocery.search", {"query": q})]
    if p["in_stock"]:
        calls.append(("grocery.add_to_cart", {"product_id": p["id"], "quantity": n}))
    return Script(c, {"grocery": st}).request(c.wrap(f"Nếu {q} còn hàng thì thêm {say_num(c.rng, n)} {p['unit']} vào giỏ, hết thì thôi"), calls)


def grocery_ask(c):
    rows, st = groc_state(c)
    q, p = need_stock(rows, c)
    t = c.rng.choice([f"Thêm {q} vào giỏ đi chợ {c.help()}", f"Cho {q} vô giỏ hàng siêu thị"])
    return Script(c, {"grocery": st}).request(c.wrap(t), [("grocery.search", {"query": q}), ("sys.ask_missing_info", {"tool_name": "grocery.add_to_cart", "params": ["quantity"]})])


def ecom_search(c):
    rows, st = ecom_state(c)
    q, p = c.rng.choice(rows)
    if c.rng.random() < 0.5:
        return Script(c, {"ecommerce": st}).request(c.wrap(f"Tìm {c.help()} {q} trên sàn"), [("ecommerce.search", {"query": q})])
    cap = (p["price"] // 100000 + 2) * 100000
    return Script(c, {"ecommerce": st}).request(c.wrap(f"Kiếm {c.help()} {q} dưới {say_money(c.rng, cap)}"), [("ecommerce.search", {"query": q, "max_price": cap})])


def ecom_buy(c):
    rows, st = ecom_state(c)
    q, p = need_stock(rows, c)
    n = c.rng.randint(1, 3)
    pay = c.rng.choice([(None, "", {}), ("ewallet", ", trả bằng ví", {"payment": "ewallet"}), ("bank", ", thanh toán qua ngân hàng", {"payment": "bank"}), ("cod", ", trả tiền khi nhận hàng", {"payment": "cod"})])
    place = c.rng.choice([("về nhà", {"address": c.home}), ("tới công ty", {"address": c.work})])
    return Script(c, {"ecommerce": st}).request(c.wrap(f"Đặt mua {c.help()} {say_num(c.rng, n)} cái {q} trên sàn, giao {place[0]}{pay[1]}"),
                                                [("ecommerce.search", {"query": q}), ("ecommerce.add_to_cart", {"product_id": p["id"], "quantity": n}), ("ecommerce.checkout", {**place[1], **pay[2]})])


def ecom_status(c):
    _, st = ecom_state(c)
    return Script(c, {"ecommerce": st}).request(c.wrap(c.rng.choice([f"Đơn hàng trên sàn của {c.me} tới đâu rồi", "Xem giúp mấy đơn online đang giao tới đâu", f"Đơn hàng của {c.me} giao chưa"])), [("ecommerce.order_status", {})])


def ecom_cancel(c):
    _, st = ecom_state(c)
    live = [o for o in st["orders"] if o["status"] in ("placed", "shipping")]
    if not live:
        return None
    o = c.rng.choice(live)
    name = o["items"][0]["name"]
    return Script(c, {"ecommerce": st}).request(c.wrap(f"Huỷ {c.help()} đơn {name}"), [("ecommerce.order_status", {}), ("ecommerce.cancel_order", {"order_id": o["id"]})])


def ecom_return(c):
    _, st = ecom_state(c)
    done = [o for o in st["orders"] if o["status"] == "delivered"]
    if not done:
        return None
    o = c.rng.choice(done)
    say, reason = c.rng.choice([("bị hỏng", "damaged"), ("giao nhầm hàng", "wrong_item"), ("không ưng nữa", "changed_mind")])
    return Script(c, {"ecommerce": st}).request(c.wrap(f"{o['items'][0]['name']} {say}, trả lại đơn đó {c.help()}"), [("ecommerce.order_status", {}), ("ecommerce.request_return", {"order_id": o["id"], "reason": reason})])


def rec_state(c):
    rows = []
    for name, servings, ing in c.rng.sample(RECIPES, c.rng.randint(3, 6)):
        rows.append({"id": c.ident("rcp"), "name": name, "servings": servings, "minutes": c.rng.choice([15, 25, 30, 45, 60, 90]), "tags": [],
                     "ingredients": [{"name": x, "quantity": c.rng.randint(1, 5) * 100, "unit": "gam"} for x in ing], "steps": ["Sơ chế", "Nấu"]})
    return rows, {"recipes": rows, "saved": []}


def recipe_search(c):
    rows, st = rec_state(c)
    r = c.rng.choice(rows)
    q = r["name"].lower()
    if c.rng.random() < 0.5:
        n = c.rng.randint(2, 8)
        return Script(c, {"recipes": st}).request(c.wrap(f"Tìm {c.help()} công thức {q} cho {say_num(c.rng, n)} người"), [("recipes.search", {"query": q, "servings": n})])
    return Script(c, {"recipes": st}).request(c.wrap(f"Món {q} nấu sao, cần những gì"), [("recipes.search", {"query": q})])


def recipe_save(c):
    rows, st = rec_state(c)
    r = c.rng.choice(rows)
    q = r["name"].lower()
    return Script(c, {"recipes": st}).request(c.wrap(f"Lưu {c.help()} công thức {q}"), [("recipes.search", {"query": q}), ("recipes.save", {"recipe_id": r["id"]})])


def recipe_save_hist(c):
    rows, st = rec_state(c)
    r = c.rng.choice(rows)
    q = r["name"].lower()
    return (Script(c, {"recipes": st}).turn(c.wrap(f"Tìm công thức {q}"), [("recipes.search", {"query": q})], f"Dạ có món {r['name']}, nấu {r['minutes']} phút ạ.")
            .request(c.wrap(c.rng.choice(["Ừ lưu món đó lại", "Được đó, lưu công thức này luôn"])), [("recipes.save", {"recipe_id": r["id"]})]))


def recipe_shopping(c):
    rows, st = rec_state(c)
    r = c.rng.choice(rows)
    q = r["name"].lower()
    calls = [("recipes.search", {"query": q})] + [("shopping_list.add", {"name": i["name"]}) for i in r["ingredients"]]
    return Script(c, {"recipes": st, "shopping_list": {"items": []}}).request(c.wrap(f"Tìm món {q} rồi thêm hết nguyên liệu vào danh sách mua sắm {c.help()}"), calls)


def pan_state(c):
    rows = [{"id": c.ident("pan"), "name": n, "stock": c.rng.randint(0, 9), "unit": u} for n, u in c.rng.sample(PANTRY, c.rng.randint(4, 8))]
    return rows, {"items": rows}


def pantry_list(c):
    rows, st = pan_state(c)
    i = c.rng.choice(rows)
    return Script(c, {"pantry": st}).request(c.wrap(c.rng.choice([f"Trong bếp còn bao nhiêu {i['name'].lower()}", f"Tủ bếp nhà {c.me} còn {i['name'].lower()} không", "Xem tủ bếp còn những gì"])), [("pantry.list_items", {})])


def pantry_use(c):
    rows, st = pan_state(c)
    have = [r for r in rows if r["stock"] > 0]
    if not have:
        return None
    i = c.rng.choice(have)
    n = c.rng.randint(1, i["stock"])
    t = c.rng.choice([f"{c.me.capitalize()} vừa dùng {say_num(c.rng, n)} {i['unit']} {i['name'].lower()}, trừ trong tủ bếp {c.help()}", f"Đánh dấu đã dùng {say_num(c.rng, n)} {i['unit']} {i['name'].lower()} trong bếp"])
    return Script(c, {"pantry": st}).request(c.wrap(t), [("pantry.list_items", {}), ("pantry.consume", {"item_id": i["id"], "quantity": n})])


def pantry_low(c):
    rows, st = pan_state(c)
    i = c.rng.choice(rows)
    limit = c.rng.randint(2, 4)
    calls = [("pantry.list_items", {})]
    if i["stock"] < limit:
        calls.append(("shopping_list.add", {"name": i["name"].lower()}))
    return Script(c, {"pantry": st, "shopping_list": {"items": []}}).request(c.wrap(f"Nếu {i['name'].lower()} trong bếp còn dưới {say_num(c.rng, limit)} {i['unit']} thì thêm vào danh sách mua sắm"), calls)


def hab_state(c):
    rows = [{"id": c.ident("habit"), "name": n, "streak": c.rng.randint(0, 20), "last_logged": (c.now - timedelta(days=c.rng.randint(1, 3))).strftime("%Y-%m-%d")} for n in c.rng.sample(HABITS, c.rng.randint(3, 5))]
    return rows, {"habits": rows, "logs": []}


def habit_list(c):
    rows, st = hab_state(c)
    h = c.rng.choice(rows)
    return Script(c, {"habit_tracker": st}).request(c.wrap(c.rng.choice([f"Chuỗi {h['name'].lower()} của {c.me} được mấy ngày rồi", "Hôm nay còn thói quen nào chưa ghi không"])), [("habit_tracker.list_habits", {})])


def habit_log(c):
    rows, st = hab_state(c)
    h = c.rng.choice([r for r in rows])
    if h["last_logged"] == c.now.strftime("%Y-%m-%d"):
        return None
    when, d = c.rng.choice([("hôm nay", c.now), ("rồi", c.now)])
    t = f"{c.me.capitalize()} {h['name'].lower()} {when}, ghi lại {c.help()}"
    return Script(c, {"habit_tracker": st}).request(c.wrap(t), [("habit_tracker.list_habits", {}), ("habit_tracker.log", {"habit_id": h["id"], "date": d.strftime("%Y-%m-%d")})])


def park_state(c):
    lots = []
    for name, addr, area in LOTS[c.city]:
        lots.append({"id": c.ident("lot"), "name": name, "address": addr, "area": area, "price_per_hour": c.rng.choice([10000, 15000, 20000, 30000]), "available_spots": c.rng.randint(1, 30)})
    sessions = []
    if c.rng.random() < 0.6:
        lot = c.rng.choice(lots)
        start = c.now - timedelta(hours=1)
        sessions.append({"id": c.ident("park"), "lot_id": lot["id"], "from_time": c.iso(start), "hours": 3, "end_time": c.iso(start + timedelta(hours=3)), "total": lot["price_per_hour"] * 3, "status": "reserved"})
    return lots, {"lots": lots, "sessions": sessions}


def parking_find(c):
    lots, st = park_state(c)
    lot = c.rng.choice(lots)
    return Script(c, {"parking": st}).request(c.wrap(f"Tìm {c.help()} bãi đỗ xe gần {lot['area']}"), [("parking.find", {"near": lot["area"]})])


def parking_reserve(c):
    lots, st = park_state(c)
    lot = c.rng.choice(lots)
    s, d = c.later(0, 3)
    h = c.rng.randint(1, 5)
    t = f"Đặt {c.help()} chỗ đỗ xe ở {lot['name'].lower().replace('bãi đỗ ', 'bãi ').replace('bãi xe ', 'bãi ')} từ {s} trong {say_num(c.rng, h)} tiếng"
    return Script(c, {"parking": st}).request(c.wrap(t), [("parking.find", {"near": lot["area"]}), ("parking.reserve", {"lot_id": lot["id"], "from_time": c.iso(d), "hours": h})])


def parking_ask(c):
    lots, st = park_state(c)
    lot = c.rng.choice(lots)
    return Script(c, {"parking": st}).request(c.wrap(f"Giữ {c.help()} một chỗ đỗ xe gần {lot['area']}"), [("parking.find", {"near": lot["area"]}), ("sys.ask_missing_info", {"tool_name": "parking.reserve", "params": ["from_time", "hours", "lot_id"]})])


def parking_end(c):
    _, st = park_state(c)
    if not st["sessions"]:
        return None
    return Script(c, {"parking": st}).request(c.wrap(c.rng.choice([f"Kết thúc phiên gửi xe {c.help()}", f"{c.me.capitalize()} lấy xe rồi, kết thúc lượt đỗ xe đi"])), [("parking.list_sessions", {}), ("parking.end_session", {"session_id": st["sessions"][0]["id"]})])


def tr_state(c):
    stops = STOPS[c.city]
    routes = [{"id": f"route_{i}", "name": f"Tuyến {n}", "type": "bus", "stops": c.rng.sample(stops, min(len(stops), 4)), "fare": 7000, "duration_min": 40, "headway_min": 15} for i, n in enumerate(c.rng.sample(range(1, 90), 3))]
    deps = []
    for r in routes:
        for stop in r["stops"]:
            for k in range(2):
                deps.append({"stop": stop, "route": r["name"], "time": (c.now + timedelta(minutes=c.rng.randint(5, 90))).strftime("%H:%M")})
    return routes, {"card_balance": c.rng.choice([20000, 45000, 100000]), "routes": routes, "departures": deps, "topups": []}


def transit_route(c):
    routes, st = tr_state(c)
    a, b = c.rng.sample(STOPS[c.city], 2)
    if c.rng.random() < 0.3:
        return Script(c, {"public_transit": st}).request(c.wrap(f"Từ nhà ra {b} đi xe buýt tuyến nào"), [("public_transit.find_routes", {"origin": c.home, "destination": b})])
    return Script(c, {"public_transit": st}).request(c.wrap(f"Đi từ {a} tới {b} bằng xe buýt thì đi sao"), [("public_transit.find_routes", {"origin": a, "destination": b})])


def transit_next(c):
    routes, st = tr_state(c)
    r = c.rng.choice(routes)
    stop = c.rng.choice(r["stops"])
    if c.rng.random() < 0.5:
        return Script(c, {"public_transit": st}).request(c.wrap(f"Chuyến tiếp theo ở {stop} mấy giờ"), [("public_transit.next_departures", {"stop": stop})])
    num = r["name"].split()[-1]
    return Script(c, {"public_transit": st}).request(c.wrap(f"Tuyến {num} ở {stop} bao lâu nữa có xe"), [("public_transit.next_departures", {"stop": stop, "route": num})])


def transit_topup(c):
    _, st = tr_state(c)
    n = c.rng.choice([20000, 50000, 100000, 200000])
    return Script(c, {"public_transit": st}).request(c.wrap(c.rng.choice([f"Nạp {c.help()} {say_money(c.rng, n)} vào thẻ xe buýt", f"Thẻ đi metro nạp thêm {say_money(c.rng, n)}"])), [("public_transit.topup_card", {"amount": n})])


def trip_state(c, kind=None):
    origin = c.city if c.rng.random() < 0.7 else c.rng.choice(ROUTES)
    dest = c.rng.choice([r for r in ROUTES if r != origin])
    day = c.now + timedelta(days=c.rng.randint(1, 9))
    trips = []
    for _ in range(c.rng.randint(2, 4)):
        ty = kind or c.rng.choice(["train", "bus"])
        h = c.rng.randint(5, 21)
        trips.append({"id": c.ident("trip"), "type": ty, "operator": c.rng.choice(OPERATORS[ty]), "origin": origin, "destination": dest, "date": day.strftime("%Y-%m-%d"),
                      "depart": f"{h:02d}:00", "arrive": f"{(h + 6) % 24:02d}:00", "price": c.rng.choice([180, 250, 320, 450, 680]) * 1000, "seats_left": c.rng.randint(2, 30)})
    return origin, dest, day, trips, {"trips": trips, "bookings": []}


def trip_search(c):
    origin, dest, day, trips, st = trip_state(c, "train")
    say = say_day(c.now, day)
    return Script(c, {"train_bus_tickets": st}).request(c.wrap(f"Xem {c.help()} tàu từ {origin} đi {dest} {say} có chuyến nào"), [("train_bus_tickets.search", {"origin": origin, "destination": dest, "date": day.strftime("%Y-%m-%d"), "type": "train"})])


def trip_book(c):
    kind = c.rng.choice(["train", "bus"])
    origin, dest, day, trips, st = trip_state(c, kind)
    pick = min(trips, key=lambda t: t["price"])
    n = c.rng.randint(1, 2)
    word = "tàu" if kind == "train" else "xe khách"
    t = f"Đặt {c.help()} {say_num(c.rng, n)} vé {word} rẻ nhất từ {origin} đi {dest} {say_day(c.now, day)}"
    return Script(c, {"train_bus_tickets": st}).request(c.wrap(t), [("train_bus_tickets.search", {"origin": origin, "destination": dest, "date": day.strftime("%Y-%m-%d"), "type": kind}), ("train_bus_tickets.book", {"trip_id": pick["id"], "seats": n})])


def trip_ask(c):
    origin, dest, *_ , st = trip_state(c)
    return Script(c, {"train_bus_tickets": st}).request(c.wrap(f"Đặt {c.help()} vé đi {dest}"), [("sys.ask_missing_info", {"tool_name": "train_bus_tickets.search", "params": ["date", "origin"]})])


def trip_cancel(c):
    origin, dest, day, trips, st = trip_state(c)
    t = trips[0]
    st["bookings"].append({"id": c.ident("tk"), "trip_id": t["id"], "seats": 1, "total": t["price"], "status": "confirmed"})
    return Script(c, {"train_bus_tickets": st}).request(c.wrap(f"Huỷ {c.help()} vé đi {dest}"), [("train_bus_tickets.my_bookings", {}), ("train_bus_tickets.cancel", {"booking_id": st["bookings"][0]["id"]})])


def ev_state(c):
    rows = []
    for name, cat, venue in c.rng.sample(EVENTS, c.rng.randint(3, 6)):
        d = c.now + timedelta(days=c.rng.randint(1, 20))
        rows.append({"id": c.ident("event"), "name": name, "category": cat, "date": d.strftime("%Y-%m-%d"), "time": "19:30", "venue": venue, "price": c.rng.choice([80, 120, 200, 350]) * 1000, "available": c.rng.randint(3, 80)})
    return rows, {"events": rows, "bookings": []}


def event_search(c):
    rows, st = ev_state(c)
    e = c.rng.choice(rows)
    q = c.rng.choice([e["name"].lower(), e["category"]])
    return Script(c, {"event_tickets": st}).request(c.wrap(f"Có sự kiện {q} nào không, tìm {c.help()}"), [("event_tickets.search", {"query": q})])


def event_book(c):
    rows, st = ev_state(c)
    e = c.rng.choice(rows)
    n = c.rng.randint(1, 3)
    return Script(c, {"event_tickets": st}).request(c.wrap(f"Đặt {c.help()} {say_num(c.rng, n)} vé {e['name'].lower()}"), [("event_tickets.search", {"query": e["name"].lower()}), ("event_tickets.book", {"event_id": e["id"], "quantity": n})])


def event_book_hist(c):
    rows, st = ev_state(c)
    e = c.rng.choice(rows)
    n = c.rng.randint(1, 3)
    return (Script(c, {"event_tickets": st}).turn(c.wrap(f"Tìm vé {e['name'].lower()}"), [("event_tickets.search", {"query": e["name"].lower()})], f"Dạ có {e['name']} ở {e['venue']}, {e['price']} đồng một vé ạ.")
            .request(c.wrap(f"Ừ lấy {say_num(c.rng, n)} vé nhé"), [("event_tickets.book", {"event_id": e["id"], "quantity": n})]))


def event_ask(c):
    rows, st = ev_state(c)
    e = c.rng.choice(rows)
    return Script(c, {"event_tickets": st}).request(c.wrap(f"Đặt {c.help()} vé {e['name'].lower()}"), [("event_tickets.search", {"query": e["name"].lower()}), ("sys.ask_missing_info", {"tool_name": "event_tickets.book", "params": ["quantity"]})])


def st_state(c):
    picks = c.rng.sample(STOCKS, c.rng.randint(3, 6))
    quotes, holdings = [], []
    for sym, name in picks:
        price = c.rng.randint(12, 130) * 1000 + c.rng.choice([0, 50, 100, 150, 200, 500])
        quotes.append({"symbol": sym, "name": name, "price": price, "change_pct": round(c.rng.uniform(-4, 4), 1)})
        if c.rng.random() < 0.6:
            holdings.append({"symbol": sym, "name": name, "quantity": c.rng.choice([100, 200, 300, 500, 1000]), "avg_price": price + c.rng.randint(-5000, 5000), "price": price})
    return quotes, holdings, {"holdings": holdings, "orders": [], "alerts": [], "quotes": quotes}


def stock_quote(c):
    q, _, st = st_state(c)
    p = c.rng.choice(q)
    t = c.rng.choice([f"Giá cổ phiếu {p['symbol']} hôm nay bao nhiêu", f"Mã {p['symbol']} đang thế nào rồi", f"Xem {c.help()} giá {p['symbol']}"])
    return Script(c, {"stocks": st}).request(c.wrap(t), [("stocks.get_quote", {"symbol": p["symbol"]})])


def stock_portfolio(c):
    _, _, st = st_state(c)
    return Script(c, {"stocks": st}).request(c.wrap(c.rng.choice([f"Danh mục chứng khoán của {c.me} đang lời hay lỗ", f"{c.me.capitalize()} đang có những cổ phiếu nào", "Xem danh mục cổ phiếu"])), [("stocks.get_portfolio", {})])


def stock_buy(c):
    q, _, st = st_state(c)
    p = c.rng.choice(q)
    n = c.rng.choice([100, 200, 300, 500])
    if c.rng.random() < 0.5:
        return Script(c, {"stocks": st}).request(c.wrap(f"Mua {c.help()} {say_num(c.rng, n)} cổ phiếu {p['symbol']}"), [("stocks.place_order", {"symbol": p["symbol"], "side": "buy", "quantity": n})])
    limit = p["price"] - p["price"] % 1000
    return Script(c, {"stocks": st}).request(c.wrap(f"Đặt lệnh mua {say_num(c.rng, n)} cổ {p['symbol']} giá {say_money(c.rng, limit)}"), [("stocks.place_order", {"symbol": p["symbol"], "side": "buy", "quantity": n, "price": limit})])


def stock_sell_all(c):
    _, h, st = st_state(c)
    if not h:
        return None
    p = c.rng.choice(h)
    return Script(c, {"stocks": st}).request(c.wrap(f"Bán hết cổ phiếu {p['name']} {c.help()}"), [("stocks.get_portfolio", {}), ("stocks.place_order", {"symbol": p["symbol"], "side": "sell", "quantity": p["quantity"]})])


def stock_if(c):
    q, _, st = st_state(c)
    p = c.rng.choice(q)
    limit = (p["price"] // 1000 + c.rng.choice([-3, 3])) * 1000
    n = c.rng.choice([100, 200, 500])
    calls = [("stocks.get_quote", {"symbol": p["symbol"]})]
    if p["price"] < limit:
        calls.append(("stocks.place_order", {"symbol": p["symbol"], "side": "buy", "quantity": n}))
    return Script(c, {"stocks": st}).request(c.wrap(f"Nếu {p['symbol']} đang dưới {say_money(c.rng, limit)} thì mua {say_num(c.rng, n)} cổ, không thì thôi"), calls)


def stock_alert(c):
    q, _, st = st_state(c)
    p = c.rng.choice(q)
    up = c.rng.random() < 0.5
    price = (p["price"] // 1000 + (5 if up else -5)) * 1000
    t = f"Báo {c.me} khi {p['symbol']} {'lên tới' if up else 'xuống dưới'} {say_money(c.rng, price)}"
    return Script(c, {"stocks": st}).request(c.wrap(t), [("stocks.set_price_alert", {"symbol": p["symbol"], "price": price, "direction": "above" if up else "below"})])


def translate(c):
    s = c.rng.choice(SENTENCES)
    name, code = c.rng.choice(LANGS)
    t = c.rng.choice([f"Câu \"{s}\" tiếng {name.split()[-1]} nói sao", f"Dịch {c.help()} câu {s} sang {name}"])
    return Script(c, {"translator": {}}).request(c.wrap(t), [("translator.translate", {"text": s, "to_lang": code})])


def contact_rows(c):
    return [{"id": c.ident("c"), "name": n, "relation": "", "phone": "09" + "".join(str(c.rng.randint(0, 9)) for _ in range(8))} for n in c.rng.sample(CONTACTS, c.rng.randint(4, 8))]


def combo_parking_remind(c):
    lots, st = park_state(c)
    lot = c.rng.choice(lots)
    s, d = c.later(1, 3)
    h = c.rng.randint(2, 4)
    remind = d + timedelta(hours=h) - timedelta(minutes=30)
    t = f"Đặt chỗ đỗ xe gần {lot['area']} từ {s} trong {say_num(c.rng, h)} tiếng, rồi nhắc {c.me} lấy xe trước lúc hết giờ nửa tiếng"
    return Script(c, {"parking": st, "reminders": {"reminders": []}}).request(c.wrap(t), [("parking.find", {"near": lot["area"]}), ("parking.reserve", {"lot_id": lot["id"], "from_time": c.iso(d), "hours": h}), ("reminders.create", {"content": "lấy xe", "time": c.iso(remind)})])


def combo_event_calendar(c):
    rows, st = ev_state(c)
    e = c.rng.choice(rows)
    start = datetime.fromisoformat(f"{e['date']}T19:30:00")
    t = f"Đặt hai vé {e['name'].lower()} rồi ghi vào lịch {c.help()}"
    return Script(c, {"event_tickets": st, "calendar": {"events": []}}).request(c.wrap(t), [("event_tickets.search", {"query": e["name"].lower()}), ("event_tickets.book", {"event_id": e["id"], "quantity": 2}), ("calendar.create_event", {"title": e["name"], "start": c.iso(start), "location": e["venue"]})])


def combo_trip_sms(c):
    origin, dest, day, trips, st = trip_state(c, "train")
    pick = min(trips, key=lambda t: t["price"])
    contacts = contact_rows(c)
    who = contacts[0]
    t = f"Đặt một vé tàu rẻ nhất đi {dest} {say_day(c.now, day)} từ {origin}, rồi nhắn SMS cho {who['name']} là {c.me} đi chuyến đó"
    return Script(c, {"train_bus_tickets": st, "contacts": {"contacts": contacts}, "sms": {"messages": []}}).request(
        c.wrap(t), [("train_bus_tickets.search", {"origin": origin, "destination": dest, "date": day.strftime("%Y-%m-%d"), "type": "train"}), ("train_bus_tickets.book", {"trip_id": pick["id"], "seats": 1}),
                    ("contacts.search", {"query": who["name"]}), ("sms.send", {"contact_id": who["id"], "content": f"{c.me} đi chuyến tàu {pick['operator']} {pick['depart']}"})])


def combo_stock_note(c):
    _, h, st = st_state(c)
    return Script(c, {"stocks": st, "notes": {"notes": []}}).request(c.wrap(f"Xem danh mục cổ phiếu rồi ghi chú tổng giá trị lại {c.help()}"),
                                                                     [("stocks.get_portfolio", {}), ("notes.create", {"title": "Danh mục cổ phiếu", "content": f"Tổng giá trị {sum(x['price'] * x['quantity'] for x in h)} đồng"})])


def not_installed(c):
    """The app that could do this is missing, so the right answer is no call at all."""
    t, used = c.rng.choice([(f"Đặt {c.help()} vé xem bóng đá tối mai", "event_tickets"), ("Mua hai trăm cổ phiếu Thép Minh Long", "stocks"), (f"Tìm bãi đỗ xe gần {c.city}", "parking"),
                            ("Dịch câu chào buổi sáng sang tiếng Nhật", "translator"), (f"Nấu món phở bò cần những gì, tìm {c.help()}", "recipes"), ("Thêm trứng vào giỏ đi chợ online", "grocery")])
    s = Script(c, {}).request(c.wrap(t), [])
    s.exclude = used
    return s


SCENARIOS = [grocery_search, grocery_add, grocery_buy, grocery_after_search, grocery_if_stock, grocery_ask, ecom_search, ecom_buy, ecom_status, ecom_cancel, ecom_return,
             recipe_search, recipe_save, recipe_save_hist, recipe_shopping, pantry_list, pantry_use, pantry_low, habit_list, habit_log, parking_find, parking_reserve, parking_ask,
             parking_end, transit_route, transit_next, transit_topup, trip_search, trip_book, trip_ask, trip_cancel, event_search, event_book, event_book_hist, event_ask,
             stock_quote, stock_portfolio, stock_buy, stock_sell_all, stock_if, stock_alert, translate, combo_parking_remind, combo_event_calendar, combo_trip_sms, combo_stock_note, not_installed]


def build_sample(script: Script, sid: str, catalog: dict) -> dict:
    c = script.ctx
    used = dict(script.states)
    extra = [a for a in DISTRACTORS if a not in used and a != getattr(script, "exclude", None)]
    installed = list(used) + c.rng.sample(extra, c.rng.randint(2, 9))
    c.rng.shuffle(installed)
    tools_of = {a: [t for t in catalog if t.startswith(a + ".")] for a in installed}
    apps_state = {a: {"tools": tools_of[a], "state": used.get(a, {})} for a in installed}
    ctx = {"current_time": c.iso(c.now), "timezone": "Asia/Ho_Chi_Minh", "location": {"lat": c.lat, "lng": c.lng},
           "user_profile": {"name": c.name, "home_address": c.home, "work_address": c.work, "city": c.city}, "installed_apps": apps_state}
    sim = Simulator({"id": sid, "context": ctx}, {"tools": list(catalog.values())})
    history = []
    for n, (text, calls, reply) in enumerate(script.history, 1):
        history.append({"role": "user", "audio": f"synth/{sid}_u{n}.wav", "audio_text": text})
        results = [sim.call(name, args) for name, args in calls]
        history.append({"role": "assistant", "text": reply, "tool_calls": [{"name": a, "arguments": b} for a, b in calls], "tool_results": results})
    results = []
    for name, args in script.final_calls:
        result = sim.call(name, args)
        if not result.get("ok") or result.get("done") is False:
            raise ValueError(f"{name} {args}: {result}")
        results.append(result)
    return {"id": sid, "audio": f"synth/{sid}.wav", "audio_text": script.final_text, "history": history,
            "context": {**ctx, "installed_apps": {a: {"tools": tools_of[a]} for a in installed}},
            "label": {"tool_calls": [{"name": a, "arguments": b} for a, b in script.final_calls], "tool_results": results}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(REPO / "src" / "lifemate" / "synth_samples.jsonl"))
    args = ap.parse_args()
    catalog = catalog_tools(json.loads((PUBLIC / "public_test_tools_catalog.json").read_text(encoding="utf-8")))
    rng = random.Random(args.seed)
    rows, failures = [], {}
    for scenario in SCENARIOS:
        made, tries = 0, 0
        while made < args.per_scenario and tries < args.per_scenario * 6:
            tries += 1
            try:
                script = scenario(Ctx(rng))
                if script is None:
                    continue
                rows.append(build_sample(script, f"synth_{len(rows):05d}", catalog))
                made += 1
            except (ValueError, TypeError, IndexError) as e:
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
