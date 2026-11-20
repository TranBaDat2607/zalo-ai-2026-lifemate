"""Synthetic request texts for method B (synthetic voices, log.md 5 Nov 2026): real train requests with their names,
places, English words and numbers swapped for others.

    python scripts/make_synth_asr.py --out package_data/synth_asr/texts.json

Every text keeps the fold of the train sample it comes from (asr_train.sample_folds: 3 folds, seed 2026). A fold model
trains only on texts of the other folds, and replacement values come from generic lists written here or from samples of
the same fold, so nothing of a held-out sample reaches its fold's training. Each text gets a voice: a train clip of the
same fold (3-8 s) to clone, or one of the TTS's preset voices. Only train.jsonl and the train audio are read.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data  # noqa: E402
from lifemate.asr_train import clips_of, sample_folds  # noqa: E402

VI_GIVEN = ("An Anh Bảo Bình Châu Chi Cường Dũng Duy Dương Đạt Điệp Giang Hà Hải Hạnh Hằng Hiếu Hoa Hòa Hoàng Hồng Huệ "
            "Hùng Hưng Hương Huy Huyền Khang Khánh Khoa Khôi Kiên Lan Lâm Liên Linh Loan Long Lộc Mai Minh My Nam Nga Ngân "
            "Ngọc Nhân Nhi Nhung Oanh Phát Phong Phúc Phương Quân Quang Quế Quỳnh Sơn Tài Tâm Thắng Thanh Thảo Thịnh Thoa "
            "Thu Thủy Thư Tiến Toàn Trang Trâm Trí Trinh Trung Tú Tuấn Tùng Tuyết Uyên Vân Việt Vinh Vũ Vy Xuân Yến Bống Tí "
            "Hàn Đức Nhật Điền Trân Hạ Mây Nguyệt Diệp Kha Lực Mẫn Nghĩa Phụng Sang Thái Thông Tín Trọng Vượng Hậu Lụa Na "
            "Ninh Khuê Ly Hiền Đào Cúc Lài Sen Tùng Bách Kiệt Lợi Phú Quý Tâm Hảo Hoài Kim").split()
VI_SURNAME = ("Nguyễn Trần Lê Phạm Hoàng Huỳnh Phan Vũ Võ Đặng Bùi Đỗ Hồ Ngô Dương Lý Đinh Trịnh Đoàn Lương Mai Tô Hà "
              "Cao Lâm Tạ Châu Trương Quách Kiều").split()
VI_MIDDLE = "Văn Thị Hữu Đức Minh Ngọc Thanh Thu Quốc Gia Bảo Hoàng Anh Kim Xuân Mỹ Công Đình Thùy Phương".split()
VI_PLACES = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Huế", "Hải Phòng", "Cần Thơ", "Nha Trang", "Đà Lạt", "Vũng Tàu",
             "Quy Nhơn", "Phú Quốc", "Hạ Long", "Sa Pa", "Hội An", "Buôn Ma Thuột", "Biên Hòa", "Thủ Đức", "Bình Dương",
             "Long An", "Bắc Ninh", "Nam Định", "Thanh Hóa", "Nghệ An", "Vinh", "Quảng Ninh", "Ba Đình", "Hoàn Kiếm",
             "Đống Đa", "Cầu Giấy", "Tây Hồ", "Hai Bà Trưng", "Láng Hạ", "Kim Mã", "Nguyễn Huệ", "Lê Lợi",
             "Trần Hưng Đạo", "Điện Biên Phủ", "Võ Văn Tần", "Phạm Ngũ Lão", "Bến Thành", "Thảo Điền", "Phú Mỹ Hưng",
             "Gò Vấp", "Tân Bình", "Bình Thạnh", "Hải Châu", "Sơn Trà", "Ngũ Hành Sơn", "Mỹ Khê", "Hồ Gươm",
             "Tràng Tiền", "Hàng Bài", "Bà Triệu", "Lý Thường Kiệt", "Nguyễn Trãi", "Tôn Đức Thắng", "Xuân Thủy",
             "Phan Xích Long", "Nam Từ Liêm", "Long Biên", "Hà Đông", "Thanh Xuân", "Quận Một", "Quận Bảy", "Cát Bà",
             "Mũi Né", "Côn Đảo", "Tam Đảo", "Ninh Bình", "Hà Giang", "Cao Bằng", "Lạng Sơn", "Tây Ninh", "Bến Tre"]
FOREIGN_GIVEN = ("Emily Kevin Sarah Grace James Sophie John David Michael Anna Lisa Tom Peter Mary Linda Jack Lucy Daniel "
                 "Emma Oliver Olivia Henry Alex Chris Ryan Jason Jessica Amy Kate Ben Sam Mark Paul Laura Rachel Steven "
                 "Tony Victor William Helen Jenny Andy Eric Frank George Harry Ivy Leo Mia Nina Oscar Rose Ted Zoe").split()
FOREIGN_SURNAME = ("Walsh Wong Smith Brown Lee Taylor Clark Miller Davis Wilson Moore Hall Young King Wright Green Baker "
                   "Adams Nelson Carter Tan Lim Chen Kim Park").split()
FOREIGN_PLACES = ["Taipei", "Kuala Lumpur", "Bangkok", "Singapore", "Tokyo", "Seoul", "Osaka", "Hong Kong", "Sydney",
                  "Melbourne", "Paris", "London", "Berlin", "New York", "Los Angeles", "Dubai", "Jakarta", "Manila",
                  "Bali", "Phuket", "Shanghai", "Busan", "Chiang Mai", "Vientiane", "Phnom Penh"]
BRAND_HEAD = ("Bright Crest Oak Sun Pine River Green Blue Silver Golden Star Moon Sky Ocean Maple Cedar Stone Lake Hill "
              "Park Red White Royal Grand Prime Smart Happy Lucky Fresh Urban Metro City Home Lotus Bamboo Jade Pearl "
              "Swift Bold Nova Zen").split()
BRAND_TAIL = ("path view wood rise point field side land way gate bridge light line care mart food flix go hub lab tech "
              "net pay bank home house garden plaza center hotel spa fit med books coffee tea kitchen express air travel").split()
BRAND_VI_HEAD = "Minh Vivu Xanh Phố Nhà Bếp Ngon Vui Mát Việt Sài Gòn An Phúc Lộc Hồng Sen Tre".split()
BRAND_WORD = "Plus Pro Max Mini Lite Premium Express Center Hotel Hotels Residence Garden Plaza Studio Store Mart".split()
ENGLISH = ("book tour app list email mail podcast workshop check online offline demo brunch homestay vendor release call "
           "chill metro taxi robot meeting deadline file slide laptop wifi password shop voucher sale order ship shipper "
           "combo menu size team project report update cancel booking room lobby gym yoga spa massage coffee pizza burger "
           "sushi steak salad smoothie cocktail game video clip live stream playlist album remix trend zoom link code web "
           "website blog post story like share comment follow inbox chat timer note task schedule event party concert show "
           "ticket flight gate terminal hotel resort villa apartment office building tower plaza mall market store outlet "
           "clinic lab test scan insurance card visa wallet cashback point member gold premium plus pro max mini basic "
           "standard family kids").split()
ACRONYMS = "SMS QR TV AC ATM OTP VIP GPS PDF USB CV HR IT KPI PIN ID VAT COD MC DJ CEO NFC HD VND USD EUR SJC".split()
PERSON_CUES = set("anh chị em cô chú bác ông bà cậu mợ dì dượng thím bạn cháu con bé mẹ bố ba má thầy sếp cho gọi nhắn "
                  "với gửi hỏi mời tên mr mrs ms".split())
PLACE_CUES = set("ở tại đến tới từ về ra vô vào qua lên xuống quận huyện phường xã đường phố tỉnh thành khu chợ sân bay "
                 "quán hàng sạn cafe café siêu thị viện trường viên hồ núi biển bến ga cầu".split())
NO_SWAP_PREV = set("tiếng nước người".split())  # languages and nationalities: "tiếng Nhật", "người Hàn"
KEEP = set("Zalo Momo MoMo Grab Shopee Tiki Lazada Viettel VNPay Gojek Facebook Messenger YouTube Youtube Spotify Netflix "
           "Google Gmail iPhone Samsung Apple Be Xanh SM".split())
CURRENCY = "euro đô yên won bảng baht".split()
COUNTRIES = set("Việt Nam;Hàn Quốc;Nhật Bản;Trung Quốc;Thái Lan;Mỹ;Anh;Pháp;Đức;Úc;Nga;Lào;Campuchia;Singapore;Malaysia;"
                "Indonesia;Philippines;Ấn Độ;Canada;Ý;Tây Ban Nha;Hà Lan".split(";"))
AIRLINES = "VN VJ QH BL VU RV".split()
ENGLISH_SWAP = 0.3  # most English words keep their place and only get a new voice, so the sentence stays natural
PRESET_VOICES = ["Anh Khôi", "Mai Anh", "Minh Quân", "Minh Đức", "Mạnh Dũng", "Ngọc Huyền", "Ngọc Linh", "Phạm Tuyên",
                 "Quỳnh Anh", "Thanh Bình", "Trúc Ly", "Xuân Vĩnh", "Đoan Trang", "Ngọc Trân", "Quang Sơn", "Adam",
                 "Kim Thanh", "Minh Triết", "Mỹ Duyên", "Thái Sơn", "Thùy Dung", "Thục Đoan", "Đức Trí"]
_VI = re.compile(r"^(b|c|ch|d|đ|g|gh|gi|h|k|kh|l|m|n|ng|ngh|nh|p|ph|q|qu|r|s|t|th|tr|v|x)?[aeiouyăâêôơư"
                 r"àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]+(c|ch|m|n|ng|nh|p|t)?$")
_STRIP = ".,?!:;…\"'()"


def is_vi(word: str) -> bool:
    return bool(_VI.match(word.lower()))


def is_ascii_foreign(word: str) -> bool:
    return word.isascii() and word.isalpha() and (word.isupper() and len(word) >= 2 or not is_vi(word))


def brand(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.5:
        return rng.choice(BRAND_HEAD) + rng.choice(BRAND_TAIL)
    if roll < 0.75:
        return rng.choice(BRAND_VI_HEAD) + rng.choice(BRAND_TAIL).capitalize()
    return rng.choice(BRAND_HEAD) + rng.choice(BRAND_TAIL) + " " + rng.choice(BRAND_WORD)


def vi_person(n_words: int, rng: random.Random) -> str:
    if n_words <= 1:
        return rng.choice(VI_GIVEN)
    if n_words == 2:
        return (rng.choice(VI_MIDDLE) if rng.random() < 0.5 else rng.choice(VI_SURNAME)) + " " + rng.choice(VI_GIVEN)
    return " ".join([rng.choice(VI_SURNAME), rng.choice(VI_MIDDLE), rng.choice(VI_GIVEN)])


def same_length(options: list, n_words: int, rng: random.Random) -> str:
    fit = [o for o in options if len(o.split()) == n_words] or options
    return rng.choice(fit)


class Pools:
    """Replacement values: the generic lists plus the names and words found in one fold's samples."""

    def __init__(self, items: list):
        self.fold_brands, self.fold_english = [], []
        for item in items:
            for _, text in clips_of(item):
                for kind, words in runs(text):
                    if kind == "cap" and any(is_ascii_foreign(w) for w in words) and " ".join(words) not in KEEP:
                        self.fold_brands.append(" ".join(words))
                    elif kind == "english":
                        self.fold_english.append(words[0])

    def replace(self, kind: str, words: list, prev: str, nxt: str, rng: random.Random):
        """The new text for a run, or None to leave it as it is."""
        n = len(words)
        run = " ".join(words)
        if kind == "number":
            return number_like(words[0], prev, nxt, rng)
        if kind == "acronym":  # SMS, SJC, QR, VTV6 keep their meaning; they only get a new voice
            return None
        if kind == "code":  # flight numbers and other codes: same shape, new letters and digits
            letters = re.match(r"[A-Z]*", run).group()
            new_letters = rng.choice(AIRLINES) if len(letters) == 2 else "".join(rng.choice("ABCDEFGHKLMNPQRSTUVX") for _ in letters)
            return new_letters + "".join(str(rng.randint(0, 9)) if c.isdigit() else c for c in run[len(letters):])
        if kind == "english":
            if rng.random() >= ENGLISH_SWAP:
                return None
            if words[0].lower() in CURRENCY:
                return rng.choice(CURRENCY)
            pick = rng.choice(self.fold_english) if self.fold_english and rng.random() < 0.3 else rng.choice(ENGLISH)
            return pick.capitalize() if words[0][0].isupper() else pick
        if prev in NO_SWAP_PREV or run in KEEP or run in COUNTRIES or any(w in KEEP for w in words):
            return None
        foreign = any(is_ascii_foreign(w) for w in words)
        if run in VI_PLACES or run in FOREIGN_PLACES:
            return same_length(FOREIGN_PLACES if run in FOREIGN_PLACES else VI_PLACES, n, rng)
        looks_person = words[0] in VI_SURNAME or words[-1] in VI_GIVEN or words[0] in FOREIGN_GIVEN
        if n == 1 and not foreign and not looks_person and prev not in PERSON_CUES | PLACE_CUES:
            return None  # a capitalised word in a title ("Ngày đầu tiên", "Khu vườn ..."), not a name
        person = prev in PERSON_CUES or looks_person or (prev not in PLACE_CUES and rng.random() < 0.6)
        if foreign:
            if person:
                return rng.choice(FOREIGN_GIVEN) + ((" " + rng.choice(FOREIGN_SURNAME)) if n >= 2 else "")
            if self.fold_brands and rng.random() < 0.3:
                return rng.choice(self.fold_brands)
            return same_length(FOREIGN_PLACES, n, rng) if prev in PLACE_CUES and rng.random() < 0.5 else brand(rng)
        return vi_person(n, rng) if person else same_length(VI_PLACES, n, rng)


def number_like(text: str, prev: str, nxt: str, rng: random.Random) -> str:
    """A number of the same shape that still reads naturally where it stands."""
    if ":" in text:  # a clock time
        return f"{rng.randint(1, 12)}:{rng.choice(['00', '15', '30', '45', '05', '10', '20', '40', '50'])}"
    digits = re.sub(r"\D", "", text)
    if not digits:
        return text
    if len(digits) <= 2 and not re.search(r"\D", text):
        if nxt in ("giờ", "h", "tiếng"):
            return str(rng.randint(1, 12))
        if prev == "kém" or nxt == "phút":
            return str(rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))
        if prev == "tháng" or nxt == "tháng" and prev in ("ngày", "mùng", "mồng"):
            return str(rng.randint(1, 12))
        if prev in ("ngày", "mùng", "mồng"):
            return str(rng.randint(1, 28))
        if prev == "tầng":
            return str(rng.randint(1, 9))
        if nxt == "độ":
            return str(rng.randint(16, 32))
    lead_zero = digits[0] == "0"
    new = "".join(str(rng.randint(0, 9)) for _ in digits)
    if not lead_zero and len(new) > 1:
        new = str(rng.randint(1, 9)) + new[1:]
    out, k = [], 0
    for c in text:  # keep separators such as 1.500.000
        if c.isdigit():
            out.append(new[k])
            k += 1
        else:
            out.append(c)
    return "".join(out)


def runs(text: str) -> list:
    """(kind, words) for the replaceable parts of a request, in order; kind in number/acronym/english/cap."""
    out, cap, first = [], [], True
    for raw in text.split():
        if raw in ("Mr.", "Mrs.", "Ms.", "Dr."):  # a title, not the end of a sentence
            if cap:
                out.append(("cap", cap))
                cap = []
            first = False
            continue
        word = raw.strip(_STRIP)
        kind = None
        if word and re.fullmatch(r"\d[\d.,:]*", word):
            kind = "number"
        elif word and re.fullmatch(r"[A-Z]{1,3}\d{2,5}", word):
            kind = "code"
        elif word and word.isascii() and word.isalnum() and word.upper() == word and 2 <= len(word) <= 5:
            kind = "acronym"
        elif word and any(c.isdigit() for c in word):
            kind = None
        elif word and word[0].isupper() and len(word) >= 2 and (not first or is_ascii_foreign(word)):
            kind = "cap"
        elif word and is_ascii_foreign(word) and len(word) >= 3:
            kind = "english"
        if kind == "cap":
            cap.append(word)
        else:
            if cap:
                out.append(("cap", cap))
                cap = []
            if kind:
                out.append((kind, [word]))
        if raw[-1] in ".,?!:;…" and cap:
            out.append(("cap", cap))
            cap = []
        first = raw[-1] in ".?!…"
    if cap:
        out.append(("cap", cap))
    return out


def substitute(text: str, pools: Pools, rng: random.Random, p: float = 0.9) -> tuple:
    """The text with each replaceable run swapped with probability ``p``, and the inserted values."""
    tokens = text.split()
    plain = [t.strip(_STRIP) for t in tokens]
    out, i, inserted = [], 0, []
    found = runs(text)
    # walk the tokens and the runs together: runs come in token order
    queue = list(found)
    while i < len(tokens):
        if queue and plain[i] == queue[0][1][0] and plain[i:i + len(queue[0][1])] == queue[0][1]:
            kind, words = queue.pop(0)
            n = len(words)
            prev = plain[i - 1].lower() if i else ""
            nxt = plain[i + n].lower() if i + n < len(plain) else ""
            new = pools.replace(kind, words, prev, nxt, rng) if rng.random() < p else None
            if new is not None:
                lead = re.match(r"^[" + re.escape(_STRIP) + r"]*", tokens[i]).group()
                tail = re.search(r"[" + re.escape(_STRIP) + r"]*$", tokens[i + n - 1]).group()
                out.append(lead + new + tail)
                inserted.append(new)
            else:
                out.extend(tokens[i:i + n])
            i += n
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out), inserted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "package_data" / "synth_asr" / "texts.json"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--second", type=float, default=0.5, help="chance of a second variant for a request with swappable parts")
    ap.add_argument("--plain", type=float, default=0.25, help="chance of a new voice for a request with nothing to swap")
    ap.add_argument("--preset-share", type=float, default=0.2)
    ap.add_argument("--per-voice", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    root_dir = data.find_root("train")
    items = data.load_split("train", root_dir)
    folds = sample_folds(items, 3, 2026)
    out = []
    for fold in range(3):
        mine = [i for i in items if folds[i["id"]] == fold]
        pools = Pools(mine)
        texts = []
        for item in mine:
            for audio, text in clips_of(item):
                has = bool(runs(text))
                copies = (1 + (rng.random() < args.second)) if has else int(rng.random() < args.plain)
                for _ in range(copies):
                    new, inserted = substitute(text, pools, rng) if has else (text, [])
                    texts.append({"text": new, "fold": fold, "source": audio, "swapped": len(inserted), "new": inserted})
        # voices: groups of texts share one reference clip of this fold (3-8 s), or a preset voice
        refs = []
        for item in mine:
            for audio, _ in clips_of(item):
                info = sf.info(str(Path(root_dir) / "train" / audio))
                if 3.0 <= info.duration <= 8.0:
                    refs.append(audio)
        rng.shuffle(texts)
        for start in range(0, len(texts), args.per_voice):
            voice = {"preset": rng.choice(PRESET_VOICES)} if rng.random() < args.preset_share else {"ref": rng.choice(refs)}
            for t in texts[start:start + args.per_voice]:
                t["voice"] = voice
        out += texts
        print(f"fold {fold}: {len(mine)} samples, {len(texts)} texts, {len(refs)} reference clips", flush=True)
    for n, t in enumerate(out):
        t["id"] = f"syn_{n:05d}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(out)} texts, {sum(t['swapped'] > 0 for t in out)} with swaps -> {args.out}")


if __name__ == "__main__":
    main()
