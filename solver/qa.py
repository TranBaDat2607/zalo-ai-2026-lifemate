"""Question answering over a loaded Biz (rule-based, rulebook defaults).

answer(biz, question, as_of) -> {'type': ..., 'answer': ...}
"""
import datetime as dt
import math
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

import periods as P
from textnorm import norm

TS = pd.Timestamp
DAY = dt.timedelta(days=1)


class Unanswerable(Exception):
    pass


class Abstain(Exception):
    pass


def rnd(x, nd=0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise Abstain('nan')
    q = Decimal(1).scaleb(-nd)
    v = Decimal(repr(round(float(x), 9))).quantize(q, rounding=ROUND_HALF_UP)
    return int(v) if nd == 0 else float(v)


def num(x):
    """Number answer: integer when integral, else 2 decimals."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise Abstain('nan')
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return rnd(x, 2)


def pct(a, b):
    if not b or (isinstance(b, float) and math.isnan(b)):
        raise Abstain('div0')
    return rnd((a - b) / b * 100, 2)


def share(a, b):
    if not b:
        raise Abstain('div0')
    return rnd(a / b * 100, 2)


def A(t, v):
    return {'type': t, 'answer': v}


# ----------------------------------------------------------------------------
# question context

REWRITES = [
    (r'\bd ?thu\b', 'doanh thu'), (r'\bdoanh so\b', 'doanh thu'), (r'\bdoang thu\b', 'doanh thu'),
    (r'\bdoanh thu thuan\b', 'doanh thu'),
    (r'\b(?:bnhiu|bnhieu|bnhui|bao nhiu|baonhieu|bao nhieo|bnh|bao nhiue|bao nhieu u|bao nhieu nhieu|bao nheu|bao nhu)\b', 'bao nhieu'),
    (r'\bbn\b', 'bao nhieu'),
    (r'\b(?:dh|d h|o do|order|bill|don\s?hang|hoa don|donhang)\b', 'don'),
    (r'\b(?:dc|dk|duowc)\b', 'duoc'), (r'\b(?:ko|hok|khog|khg)\b', 'khong'),
    (r'(?<!hoa )(?<!mui )(?<!mat )(?<!do )\bhong\b(?= (?:biet|co|ban|duoc|nhi|ta|ha|a)\b)', 'khong'),
    (r'\bxo voi\b', 'so voi'), (r'\bxo vs\b', 'so voi'), (r'\btang dam\b', 'tang giam'), (r'\btang giam\b', 'tang giam'),
    (r'\bnuot xe\b', 'luot xe'), (r'\bluot\b', 'luot'),
    (r'\btrc\b', 'truoc'), (r'\bvs\b', 'voi'), (r'\bmik\b', 'minh'), (r'\bphan tram\b', 'pct'),
    (r'\bmi\b(?= don)', 'may'), (r'\btb\b', 'trung binh'), (r'\b(?:binh quan|bin quan|bq)\b', 'trung binh'),
    (r'\bcu\b(?= tro len| tro xuong| tro nen)', 'trieu'), (r'\bnhut\b', 'nhat'), (r'\bnhứt\b', 'nhat'),
    (r'\bth\b(?= \d)', 'thang'), (r'\bsp\b', 'san pham'), (r'\bsl\b', 'so luong'),
    (r'\bhnay\b', 'hom nay'), (r'\bhqua\b', 'hom qua'), (r'\btong cong\b', 'tong cong'),
    (r'\bz\b', ' '), (r'\bzay\b', 'vay'), (r'\bzi\b', 'gi'), (r'\brevenue\b', 'doanh thu'),
    (r'\borde\b', 'don'), (r'\bnhiu\b', 'nhieu'), (r'\bnhiue\b', 'nhieu'), (r'\b(?:k|hk|kh|hong|ko|khong) ke don\b', 'khong ke don'),
    (r'\btien tuoi\b', 'tien mat'), (r'\bwa\b', 'qua'), (r'\bngui\b', 'nguoi'), (r'\bmi (?=nguoi|khach|ky|don|cai|may)', 'may '),
    (r'\bdong (?=tien hang|co tien hang|co doanh thu|doanh thu tu)', 'don '), (r'\bchong\b(?= (?:it nhat|toi thieu|\d|thang|tuan|quy|nam))', 'trong'),
    (r'\bnv\b(?= nao| ban| ma)', 'nhan vien'), (r'\bnh(?:ta|aat|att|aht)\b', 'nhat'), (r'\bngaynao\b', 'ngay nao'),
    (r'\bthangnao\b', 'thang nao'), (r'\bcaonhat\b', 'cao nhat'), (r'\b(?:trung bih|trug binh|trung bin|chung binh)\b', 'trung binh'),
    (r'\bgia tri (?:1|mot|moi) don\b', 'trung binh mot don gia tri'), (r'\b(?:duco|duowc|duocj|dduoc)\b', 'duoc'),
    (r'\bbao nh(?:eiu|ieeu|iue|ieu u|ieuu|iu|eu|ue|ieo|ieu)\b', 'bao nhieu'), (r'\bbao nhieu ten\b', 'bao nhieu tien'),
    (r'\btieng\b', 'tien'), (r'\btien bang\b(?= bao nhieu| nhieu)', 'tien ban'), (r'\bdoanhthu\b', 'doanh thu'),
    (r'\bvau cho\b', 'voucher'), (r'\bvoucher\b', 'vouche'), (r'\bnhan dien\b', 'nhan vien'),
    (r'\b(?:xi|si) o di\b', 'cod'), (r'\blen xuong\b', 'len hay xuong'), (r'\bxuong len\b', 'len hay xuong'),
    (r'\bdau nam (?:toi|den|cho den) (?:chu|nay|gio|bay gio|hom nay|bua nay)\b', 'tu dau nam'),
    (r'\bthang (muoi mot|muoi hai|mot|hai|ba|bay|tam|chin|muoi|(?:tu|nam|sau)(?= (?:nam )?\d{4}\b))\b(?! (?:nay|truoc|roi|ngoai|sau|toi)\b)',
     lambda m: 'thang ' + str({'mot': 1, 'hai': 2, 'ba': 3, 'tu': 4, 'nam': 5, 'sau': 6, 'bay': 7, 'tam': 8, 'chin': 9,
                                'muoi': 10, 'muoi mot': 11, 'muoi hai': 12}[m.group(1)])),
    (r'\bth\b(?= (?:truoc|nay|roi|sau|ni|trc|lien|vua)\b)', 'thang'),
    (r'\btu (?:ngay )?(?:1 1|01 01)(?: (?:nam nay|\d{4}))? (?:den|toi)(?: het)?(?: ngay)? (?:hom nay|bua nay|nay|hnay)', 'tu dau nam'),
]
CMP_PREV = (r'\b(?:so (?:voi )?|hon hay kem |cao hay thap hon |thap hay cao hon |chenh |so sanh voi |cung ky |voi )'
            r'(?:cung ky )?(?:thang|tuan|ngay|quy|nam|ky) ?(?:lien |ngay )?(?:truoc|roi|ngoai|trc)(?: do| lien ke| nua)?\b'
            r'|\bcung ky (?:thang|tuan|nam|quy) (?:truoc|roi|ngoai)\b|\b(?:thang|quy|tuan) lien truoc\b'
            r'|\b(?:thang|quy|tuan) truoc do\b|\bky truoc\b|\bcung ky\b'
            r'|\b(?:so (?:voi )?)?cung so ngay (?:dau )?(?:cua )?(?:thang|tuan) (?:truoc|roi|trc)\b')
YOY = r'cung ky nam (?:ngoai|truoc|roi)|so voi nam (?:ngoai|truoc|roi)|(?<!tet )nam ngoai|cung thang nam (?:ngoai|truoc|roi)|thang do nam (?:ngoai|truoc|roi)'


class Q:
    def __init__(self, biz, text, today):
        self.biz = biz
        self.raw = text
        self.today = today
        self.n0 = ' ' + norm(text) + ' '
        n = norm(re.sub(r'(?<![\d,.])\s*%', ' phan tram ', text).replace('≥', ' tu ').replace('>=', ' tu '))
        for a, b in REWRITES:
            n = re.sub(a, b, n)
        n = re.sub(r'\s+', ' ', n).strip()
        # 'tu dau thang nay den het hom nay' == thang nay
        n = re.sub(r'\btu dau thang (?:nay|ni)(?: (?:den|toi|cho den)(?: het)?(?: ngay)? (?:hom nay|bua nay|nay|hom ni))?', 'thang nay', n)
        n = re.sub(r'\btu dau tuan (?:nay|ni)(?: (?:den|toi)(?: het)?(?: ngay)? (?:hom nay|bua nay|nay))?', 'tuan nay', n)
        n = re.sub(r'(?<!tet )(?<!tet am lich )\b(?:tu dau nam|nam nay|nam ni)(?: nay)?(?: (?:tinh )?(?:den|toi|cho den)(?: het)?(?: ngay)? (?:hom nay|bua nay|nay|hien tai|gio))?', 'tu dau nam', n)
        n = re.sub(r'\btu dau nam tinh tu dau nam\b', 'tu dau nam', n)
        self.n = ' ' + n + ' '
        self.yoy = bool(re.search(YOY, self.n))
        self.cmp_prev = bool(re.search(CMP_PREV, self.n))
        n2 = re.sub(CMP_PREV, ' ', self.n)
        if self.yoy:
            n2 = re.sub(YOY, ' ', n2)
        self.n_nocmp = n2
        ps = []
        for _, p in P.find_periods(n2, today):
            if not any(p.start == x.start and p.end == x.end for x in ps):
                ps.append(p)
        if len(ps) >= 2:
            ps2 = [p for p in ps if not (p.kind == 'day' and p.start == today and
                                          any(x is not p and x.kind != 'day' for x in ps))]
            ps = ps2 or ps
        self.periods = ps
        # 'cac thang 10, 11, 12 nam 2025'
        m = re.search(r'\bthang ((?:\d{1,2} (?:va )?){1,11}\d{1,2}) (?:nam )?(\d{4})\b', self.n_nocmp)
        if m and len(self.periods) <= 1:
            nums = [int(x) for x in re.findall(r'\d{1,2}', m.group(1))]
            if len(nums) >= 2 and all(1 <= x <= 12 for x in nums):
                self.periods = [P.month_period(int(m.group(2)), x) for x in nums]
        # 'N thang lien nhau ket thuc o thang M/Y'
        m = re.search(r'\b(\d{1,2}) thang (?:lien nhau |lien tiep |gan nhat |lien ke |lien )?(?:ket thuc|den het|cuoi cung la|tinh toi|tinh den|toi|den)'
                      r' (?:o |vao |tai |la )?(?:thang ?|t)?(\d{1,2}) (\d{4})\b', self.n)
        if m:
            k, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 1 < k <= 24:
                self.periods = [P.month_period(*P.add_months(y, mo, -i)) for i in range(k - 1, -1, -1)]

    def has(self, rx):
        return re.search(rx, self.n) is not None

    def find(self, rx):
        return re.search(rx, self.n)


# ----------------------------------------------------------------------------
# data access

def data_range(biz):
    o = biz.orders
    start = o['date'].min().date()
    end = biz.data_end or o['date'].max().date()
    return start, end


def window(q, p, clip_today=True):
    s, e = p.start, p.end
    if clip_today:
        e = min(e, q.today)
    ds, de = data_range(q.biz)
    if e < ds or s > de or s > e:
        raise Unanswerable('period outside data')
    return max(s, ds), min(e, de)


def orders_in(q, p, valid=True):
    a, b = window(q, p)
    o = q.biz.orders
    m = (o['date'] >= TS(a)) & (o['date'] <= TS(b))
    if valid:
        m &= o['valid']
    return o[m]


def need_lines(q):
    if q.biz.lines is None:
        raise Unanswerable('no line items')
    return q.biz.lines


def lines_of(q, o):
    l = need_lines(q)
    return l[l['order_id'].isin(set(o['order_id']))]


def goods_lines(l):
    return l[~l['is_fee']]


# ----------------------------------------------------------------------------
# capabilities

def cap_time(biz):
    return biz.orders['dt'].notna().mean() > 0.5


def cap_customers(biz):
    o = biz.orders
    return o['cust_key'].notna().any()


def cap_col(biz, col):
    o = biz.orders
    return col in o and o[col].notna().mean() > 0.3


# ----------------------------------------------------------------------------
# entity matching

def product_matches(q, text=None):
    biz = q.biz
    cache = getattr(biz, '_pm', None)
    if cache is None:
        cache = []
        for k, names in (getattr(biz, 'prod_spellings', {}) or {}).items():
            for nm in set(list(names) + [biz.prod_names.get(k, '')]):
                nn = norm(nm)
                if len(nn) >= 3:
                    cache.append((nn, k))
        if biz.products is not None and 'aliases' in biz.products:
            for k, al in zip(biz.products['prod_key'], biz.products['aliases']):
                if isinstance(al, str):
                    for a in al.split(','):
                        nn = norm(a)
                        if len(nn) >= 3:
                            cache.append((nn, k))
        cache = [(syn(nn), k) for nn, k in cache]
        cache.sort(key=lambda x: -len(x[0]))
        biz._pm = cache
    s = syn(text if text is not None else q.n0)
    found, taken = [], []
    for nn, k in cache:
        for m in re.finditer(r'(?<![a-z0-9])' + re.escape(nn) + r'(?![a-z0-9])', s):
            if any(m.start() < e and m.end() > s0 for s0, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((k, m.start(), m.end()))
    if not found and text is None:
        found = _fuzzy_products(cache, s)
    found.sort(key=lambda x: x[1])
    out, seen = [], set()
    for k, a, b in found:
        if k not in seen:
            seen.add(k)
            out.append((k, a, b))
    return out


SYNONYMS = [(r'\blac\b(?= rang| luoc| hat| say| muoi| nhan)', 'dau phong'), (r'\bxay trat\b', 'xay to'),
            (r'\bcoc sac\b', 'cu sac'), (r'\bcoc\b', 'ly'), (r'\bke (?=dien thoai|dt\b)', 'gia do '), (r'\bxa bong\b', 'xa phong'),
            (r'\bmy (?=tom|goi|hao|omachi|3 mien)', 'mi '), (r'\bngo\b(?= rang| ngot| my| xao| nuong| luoc)', 'bap'),
            (r'\bdien thoai cu\b', 'may cu'), (r'\bdien thoai moi\b', 'may moi'),
            (r'\bgang tay\b', 'bao tay'), (r'\btat chan\b', 'bao chan'), (r'\bhoa qua\b', 'trai cay'),
            (r'\bdau phu\b', 'dau hu'), (r'\bran\b', 'chien'), (r'\b(thit|ma|suon|gio|chan|tai|long|nem|cha|tim|dui|ba chi|oc|luoi) lon\b', r'\1 heo'),
            (r'\bngao\b', 'ngheu'), (r'\bca (?:trau|qua)\b', 'ca loc'), (r'\bdua chuot\b', 'dua leo'), (r'\bmuop dang\b', 'kho qua'),
            (r'\bvung\b', 'me'), (r'\bthia\b', 'muong'), (r'\bmi chinh\b', 'bot ngot'), (r'\bxi dau\b', 'nuoc tuong'),
            (r'\bhot (?=vit|ga|cut)', 'trung '), (r'\bbim\b', 'ta'), (r'\blop xe\b', 'vo xe'), (r'\bbanh phu thong\b', 'banh'),
            (r'\bkem danh rang\b', 'kem danh rang'), (r'\bnuoc ngot\b', 'nuoc ngot'), (r'\bsua bot\b', 'sua bot')]


def syn(s):
    for a, b in SYNONYMS:
        s = re.sub(a, b, s)
    return s


def _fuzzy_products(cache, s):
    """Names written with one different word ('coc sac' for 'cu sac'): names of 4+ words,
    the differing word not a number / size token, unique best candidate."""
    toks = s.split()
    pos = []
    i = 0
    for t in toks:
        j = s.index(t, i)
        pos.append(j)
        i = j + len(t)
    hits = {}
    for nn, k in cache:
        w = nn.split()
        n = len(w)
        if n < 4:
            continue
        for st in range(0, len(toks) - n + 1):
            win = toks[st:st + n]
            diff = [a for a, b in zip(w, win) if a != b]
            if len(diff) == 1 and not re.search(r'\d', diff[0]) and w[0] == win[0] or \
                    len(diff) == 1 and not re.search(r'\d', diff[0]) and w[-1] == win[-1] and w[1] == win[1]:
                hits.setdefault(k, (pos[st], pos[st + n - 1] + len(toks[st + n - 1])))
    if len(hits) == 1:
        k, (a, b) = next(iter(hits.items()))
        return [(k, a, b)]
    return []


def categories(biz):
    if biz.products is None or 'category' not in biz.products:
        return []
    vals = {x for x in biz.products['category'].dropna().unique() if str(x).strip()}
    return sorted(vals, key=lambda s: -len(norm(s)))


def category_match(q):
    cats = categories(q.biz)
    hits = []
    for c in cats:
        nc = norm(c)
        for m in re.finditer(r'(?<![a-z0-9])(?:nhom |loai |nganh |danh muc )?' + re.escape(nc) + r'(?![a-z0-9])', q.n0):
            if any(m.start() < e and m.end() > s for _, s, e in hits):
                continue
            hits.append((c, m.start(), m.end()))
    hits.sort(key=lambda x: x[1])
    return hits


def cat_of(biz):
    p = biz.products
    if p is None or 'category' not in p:
        return {}
    return dict(zip(p['prod_key'], p['category']))


UNIT_WORDS = ['ly', 'lon', 'chai', 'goi', 'bao', 'cay', 'kg', 'binh', 'thung', 'hop', 'vien', 'vi', 'tuyp', 'bich',
              'cai', 'dia', 'phan', 'm', 'm2', 'm3', 'cuon', 'tam', 'bo', 'loc', 'qua', 'con', 'to', 'chen', 'nồi',
              'noi', 'xuat', 'suat', 'ong', 'lo', 'mieng', 'doi', 'set', 'chiec', 'may', 'thanh', 'gam', 'lit', 'can',
              'ket', 'xe', 'khay', 'phan', 'tui', 'li', 'coc', 'dia', 'dua', 'cap', 'mo', 'goi', 'banh', 'cuc', 'bo']


# ----------------------------------------------------------------------------
# filters: order-level and line-level

CHANNELS = [
    (r'zalo', 'zalo'), (r'shopee', 'shopee'), (r'grab', 'grab'), (r'tiktok', 'tiktok'), (r'lazada', 'lazada'),
    (r'facebook|fb\b|fanpage', 'facebook'), (r'dien thoai|goi dien|qua phone|phone', 'phone'),
    (r'tai quay|tai cua hang|tai tiem|tai quan|truc tiep|store|tai cho|tai shop', 'store'),
    (r'san thuong mai dien tu|san tmdt|san\b|thuong mai dien tu|tmdt', 'marketplace'),
    (r'online|web', 'online'), (r'giao hang tan noi|ship tan noi', 'delivery'),
]
PAYS = [
    (r'tien mat|\btm\b|cash|tien tuoi', 'cash'), (r'chuyen khoan|\bck\b|transfer|bank', 'transfer'),
    (r'vi dien tu|\bvi\b|e wallet|ewallet|momo|zalopay|zalo pay|vnpay', 'ewallet'),
    (r'\bcod\b|thu ho|giao hang thu tien|\b(?:xi|si) o di\b', 'cod'), (r'ghi no|mua chiu|ban chiu|tra sau|\bno\b|debt', 'debt'),
    (r'quet the|the ngan hang|the tin dung|the atm|\bcard\b|\bpos\b', 'card'),
]


def plain(v):
    from textnorm import strip_accents
    s = strip_accents(str(v).lower())
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', s).split())


def channel_key(v):
    s = plain(v) if v is not None else ''
    if not s:
        return None
    for rx, k in CHANNELS:
        if re.search(rx, s):
            return k
    return s


def pay_key(v):
    s = plain(v) if v is not None else ''
    if not s:
        return None
    for rx, k in PAYS:
        if re.search(rx, s):
            return k
    return s


def channel_mentions(q):
    """Ordered list of channel keys mentioned (after the word 'qua'/'tai'/'ban'/'tren')."""
    out = []
    for rx, k in CHANNELS:
        for m in re.finditer(r'\b(?:qua|tai|tren|ban|kenh|nguon|don|dat|o)\s+(?:kenh |nguon |app |ung dung )?(?:' + rx + r')',
                             q.n0 + ' | ' + re.sub(r'\bwa\b', 'qua', q.n0)):
            out.append((m.start(), k))
    out.sort()
    res = []
    for _, k in out:
        if k not in res:
            res.append(k)
    return res


def pay_mentions(q):
    out = []
    for rx, k in PAYS:
        if k == 'debt':
            rx = r'ghi no|mua chiu|ban chiu|tra sau'
        for m in re.finditer(r'\b(?:tra bang|thanh toan bang|thanh toan qua|tra qua|bang|tra|thanh toan|hinh thuc|qua|dung|don|kieu)\s+(?:tien\s+)?(?:' + rx + r')', q.n0 + ' ' + q.n):
            out.append((m.start(), k))
    out.sort()
    res = []
    for _, k in out:
        if k not in res:
            res.append(k)
    return res


def shift_of(ts):
    if ts is None or pd.isna(ts):
        return None
    h = ts.hour
    if 6 <= h < 14:
        return 'sang'
    if 14 <= h < 22:
        return 'chieu'
    return None


def shift_mentions(q):
    return [m.group(1) for m in re.finditer(r'\bca (sang|chieu)\b', q.n)]


# ----------------------------------------------------------------------------
# metric computations on (orders, lines)

def m_revenue(o):
    return float(o['rev'].sum())


_HANDLERS = None


def all_handlers():
    global _HANDLERS
    if _HANDLERS is None:
        import handlers
        import handlers2
        import handlers_special
        hs = list(handlers.HANDLERS)
        i = hs.index(handlers.h_subjective) + 1
        h2 = list(handlers2.HANDLERS2)
        first = [h for h in h2 if h.__name__ in ('h_price_change',)]
        _HANDLERS = hs[:i] + first + list(handlers_special.HANDLERS_SPECIAL) + [h for h in h2 if h not in first] + hs[i:]
    return _HANDLERS


def answer(biz, text, today):
    q = Q(biz, text, today)
    for h in all_handlers():
        try:
            r = h(q)
        except Unanswerable as e:
            return {'type': 'unanswerable', 'reason': str(e), '_h': h.__name__}
        except Abstain as e:
            return {'type': 'abstain', '_h': h.__name__, '_why': str(e)}
        if r is not None:
            r['_h'] = h.__name__
            return r
    return {'type': 'abstain', '_h': 'none'}
