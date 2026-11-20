"""Intent handlers. Each takes a Q and returns an answer dict or None."""
import datetime as dt
import re
from collections import Counter

import numpy as np
import pandas as pd

import periods as P
from qa import (A, Abstain, DAY, TS, Unanswerable, cap_col, cap_customers, cap_time, cat_of, categories,
                category_match, channel_key, channel_mentions, data_range, goods_lines, lines_of, need_lines, num,
                orders_in, pay_key, pay_mentions, pct, product_matches, rnd, share, shift_mentions, shift_of, window)
from textnorm import norm

HOW = r'(?:bao nhieu|may|so|dem|tong so|tong cong|nhieu)'
SUPER_MAX = r'(?:nhieu nhat|cao nhat|lon nhat|dat nhat|top|hot nhat|chay nhat|nhat)'


def main_period(q):
    return q.periods[0] if q.periods else None


def need_period(q):
    p = main_period(q)
    if p is None:
        raise Abstain('no period')
    return p


# ----------------------------------------------------------------------------
# money / numbers in the raw question

def money_values(q):
    """Money amounts written in the question, in VND."""
    s = q.raw.lower()
    out = []
    for m in re.finditer(r'(\d+)\s*tr\s*(\d{1,3})(?![\w])', s):
        out.append((m.start(), int(m.group(1)) * 1_000_000 + int(m.group(2).ljust(3, '0')) * 1000, 'tr'))
    for m in re.finditer(r'(\d[\d.,\s]*\d|\d)\s*(đồng|dong|đ|d|vnd|k|nghìn|ngàn|ngan|tr|triệu|trieu|củ|cu|m)?(?![\w])', s):
        numtxt, unit = m.group(1).strip(), m.group(2)
        if any(abs(p - m.start()) < 3 for p, _, u in out if u == 'tr'):
            continue
        from values import parse_number
        if unit in ('k', 'nghìn', 'ngàn', 'ngan'):
            v = parse_number(numtxt + 'k')
        elif unit in ('tr', 'triệu', 'trieu', 'củ', 'cu', 'm'):
            v = parse_number(numtxt + 'tr')
        else:
            v = parse_number(numtxt)
        if v is None:
            continue
        out.append((m.start(), v, unit))
    return out


def threshold(q):
    """(op, value) for 'tu X tro len', 'it nhat X', 'tren X', 'duoi X'."""
    n = q.n
    m = re.search(r'\b(?:tu|dat|it nhat|toi thieu|>=|tren|hon|qua|duoi|khong qua|nho hon|lon hon|tro len)\b', n)
    return m


def parse_money_threshold(q):
    """Money threshold in the raw question with its operator."""
    raw = q.raw.lower()
    vals = [v for v in money_values(q) if (v[1] >= 1000 or v[2] in ('k', 'tr', 'triệu', 'củ', 'cu', 'nghìn', 'ngàn'))
            and not (v[2] is None and 2015 <= v[1] <= 2035 and float(v[1]).is_integer())]
    vals = [v for v in vals if not re.match(r'\s*/', raw[v[0] + len(str(int(v[1]))):][:2])]
    if not vals:
        return None
    pos, v, unit = vals[-1]
    before = norm(raw[max(0, pos - 25):pos])
    after = norm(raw[pos:pos + 40])
    op = '>='
    if re.search(r'(?:tren|hon|qua|lon hon|vuot)\s*$', before) and not re.search(r'tro len', after):
        op = '>'
    if re.search(r'(?:duoi|nho hon|it hon)\s*$', before):
        op = '<'
    if re.search(r'tro xuong|do lai|tro lai', after):
        op = '<='
    return op, v


def cmp(series, op, v):
    if op == '>=':
        return series >= v - 1e-9
    if op == '>':
        return series > v + 1e-9
    if op == '<':
        return series < v - 1e-9
    return series <= v + 1e-9


def count_threshold(q, word):
    """'tu 4 don tro len' -> ('>=', 4)."""
    n = q.n
    m = re.search(r'(?:tu|it nhat|toi thieu|>=|tren|hon|qua|duoi|nhieu hon|it hon)\s+(\d+)\s+' + word, n)
    if not m:
        m = re.search(r'(\d+)\s+' + word + r'\s+(?:tro len|tro xuong)', n)
        if not m:
            return None
        v = int(m.group(1))
        return ('>=' if 'tro len' in m.group(0) else '<='), v
    v = int(m.group(1))
    w = n[m.start():m.start() + 12]
    after = n[m.end():m.end() + 10]
    if w.startswith(('tren', 'hon', 'qua', 'nhieu hon')) and 'tro len' not in after:
        return '>', v
    if w.startswith(('duoi', 'it hon')):
        return '<', v
    if 'tro xuong' in after:
        return '<=', v
    return '>=', v


# ----------------------------------------------------------------------------
# filters

class Flt:
    def __init__(self):
        self.products = []
        self.category = None
        self.channels = []
        self.pays = []
        self.shifts = []
        self.group = None
        self.staff = None
        self.rx = None
        self.size = None
        self.unit = None
        self.ingredient = None

    def line_level(self):
        return bool(self.products or self.category or self.rx is not None or self.size or self.unit or self.ingredient)

    def any(self):
        return self.line_level() or bool(self.channels or self.pays or self.shifts or self.group or self.staff)


def cust_groups(biz):
    o = biz.orders
    if 'cust_group' not in o:
        return []
    return sorted({g for g in o['cust_group'].dropna().unique()}, key=lambda s: -len(norm(s)))


def staff_list(biz):
    o = biz.orders
    ids = sorted({s for s in o['staff'].dropna().unique()}) if 'staff' in o else []
    names = biz.extra.get('staff', {}) or {}
    return ids, names


def parse_filters(q, metric=None):
    f = Flt()
    biz = q.biz
    pm = product_matches(q)
    f.products = [k for k, _, _ in pm]
    spans = [(a, b) for _, a, b in pm]
    for c, a, b in category_match(q):
        if any(a >= s and b <= e for s, e in spans):
            continue
        pre = q.n0[max(0, a - 12):a]
        if re.search(r'(?:nhom|loai|nganh|danh muc)\s*(?:hang\s*)?$', pre) or q.n0[a:b].startswith(('nhom', 'loai')):
            f.category = c
            break
    f.channels = channel_mentions(q)
    f.pays = pay_mentions(q)
    f.shifts = shift_mentions(q)
    for g in cust_groups(biz):
        ng = norm(g)
        if re.search(r'(?:nhom khach|khach nhom|khach hang nhom|khach|nhom)\s+(?:hang\s+)?' + re.escape(ng) + r'\b', q.n):
            f.group = g
            break
    m = re.search(r'nhan vien (?:ma )?([a-z]{1,3}\s?\d{2,4})\b', q.n) or re.search(r'\b(nv\s?\d{2,4})\b', q.n)
    ids, names = staff_list(biz)
    if m:
        cand = m.group(1).replace(' ', '').upper()
        if cand in ids:
            f.staff = cand
    if f.staff is None and q.has(r'nhan vien|\bnv\b|tho\b'):
        for sid, nm in names.items():
            if nm and re.search(r'\b' + re.escape(norm(nm)) + r'\b', q.n):
                f.staff = sid
                break
    if q.has(r'khong ke don|hok ke don|ko ke don|otc|khong can don|ngoai thuoc ke don|tru thuoc ke don'):
        f.rx = False
    elif q.has(r'ke don|theo don|can don'):
        f.rx = True
    m = q.find(r'\bsize ([ml])\b')
    if m:
        f.size = m.group(1).upper()
    if q.has(r'hoat chat|thanh phan'):
        p = biz.products
        if p is not None and 'ingredient' in p:
            ings = sorted({norm(x) for x in p['ingredient'].dropna() if str(x).strip()}, key=lambda s: -len(s))
            for ing in ings:
                if re.search(r'(?:hoat chat|thanh phan)(?: chinh)?(?: la)? ' + re.escape(ing) + r'(?![a-z0-9])', q.n0):
                    f.ingredient = ing
                    break
            if f.ingredient is None:
                f.ingredient = '__none__'
    return f


def unit_asked(q):
    """Unit word in 'bao nhieu <unit>' / 'may <unit>' / 'so <unit>'."""
    biz = q.biz
    units = set(biz.extra.get('base_unit', {}).values())
    for fu in biz.units.values():
        units |= set(fu.keys())
    units = {u for u in units if u}
    m = re.search(r'\b(?:bao nhieu|may|so|nhieu)\s+([a-z0-9]+(?: [a-z0-9]+)?)', q.n)
    cands = []
    for m in re.finditer(r'\b(?:bao nhieu|may|so|nhieu|tong so)\s+([a-z0-9]+)(?: ([a-z0-9]+))?(?: ([a-z0-9]+))?', q.n):
        w = [x for x in m.groups() if x]
        for k in range(len(w), 0, -1):
            u = ' '.join(w[:k])
            if u in units:
                cands.append(u)
                break
    return cands[0] if cands else None


def apply_order_filters(q, o, f):
    biz = q.biz
    if f.channels and len(f.channels) == 1:
        o = o[o['channel_k'] == f.channels[0]] if 'channel_k' in o else _no_data('channel')
    if f.pays and len(f.pays) == 1:
        o = o[o['pay_k'] == f.pays[0]] if 'pay_k' in o else _no_data('payment')
    if f.shifts and len(f.shifts) == 1:
        o = o[o['shift'] == f.shifts[0]]
    if f.group:
        o = o[o['cust_group'].map(norm) == norm(f.group)]
    if f.staff:
        o = o[o['staff'] == f.staff]
    return o


def _no_data(what):
    raise Unanswerable(f'no {what} data')


def prep_orders(biz):
    """Derived order columns (cached on biz)."""
    if getattr(biz, '_prep', False):
        return
    o = biz.orders
    if 'channel' in o and o['channel'].notna().mean() > 0.3:
        o['channel_k'] = o['channel'].map(channel_key)
    if 'pay_method' in o and o['pay_method'].notna().mean() > 0.3:
        o['pay_k'] = o['pay_method'].map(pay_key)
    o['shift'] = o['dt'].map(shift_of) if o['dt'].notna().any() else None
    biz._prep = True


def check_caps(q, f):
    biz = q.biz
    prep_orders(biz)
    o = biz.orders
    if f.channels and 'channel_k' not in o:
        raise Unanswerable('no channel data')
    if f.pays and 'pay_k' not in o:
        raise Unanswerable('no payment data')
    if f.shifts and not cap_time(biz):
        raise Unanswerable('no time of day')
    if f.group is None and q.has(r'nhom khach|khach nhom|khach (?:si|vip|than thiet|quen)\b') and not cust_groups(biz):
        raise Unanswerable('no customer groups')
    if (f.category is None and q.has(r'\bnhom\b') and not q.has(r'nhom khach|khach nhom')
            and not categories(biz) and not f.products):
        raise Unanswerable('no categories')
    if f.rx is not None and ('rx' not in biz.products or biz.products['rx'].isna().all()):
        raise Unanswerable('no prescription info')


def apply_line_filters(q, l, f):
    biz = q.biz
    if f.products:
        l = l[l['prod_key'].isin(f.products)]
    if f.category:
        cm = cat_of(biz)
        nc = norm(f.category)
        cat = l['prod_key'].map(cm)
        if 'category_line' in l and l['category_line'].notna().any():
            cat = cat.where(cat.notna(), l['category_line'])
        l = l[cat.map(lambda c: norm(c) == nc if isinstance(c, str) else False)]
    if f.rx is not None:
        rx = dict(zip(biz.products['prod_key'], biz.products['rx']))
        l = l[l['prod_key'].map(lambda k: bool(rx.get(k)) == f.rx)]
    if f.size:
        nm = biz.prod_names
        l = l[l['prod_key'].map(lambda k: str(nm.get(k, k)).strip().lower().endswith('(' + f.size.lower() + ')'))]
    if f.ingredient:
        ing = {k: norm(v) for k, v in zip(biz.products['prod_key'], biz.products['ingredient']) if isinstance(v, str)}
        l = l[l['prod_key'].map(lambda k: ing.get(k) == f.ingredient)]
    if f.unit:
        bu = biz.extra.get('base_unit', {})
        l = l[l['prod_key'].map(lambda k: bu.get(k) == f.unit)]
    return l


def metric_value(q, p, metric, f, o=None):
    biz = q.biz
    prep_orders(biz)
    if o is None:
        o = orders_in(q, p)
    o = apply_order_filters(q, o, f)
    if metric in ('revenue', 'orders', 'aov', 'qty') and (f.line_level() or metric == 'qty'):
        l = lines_of(q, o)
        if metric == 'qty' or f.products or f.category or f.size or f.unit:
            l = goods_lines(l)
        l = apply_line_filters(q, l, f)
        if metric == 'revenue':
            return float(l['amount'].sum())
        if metric == 'orders':
            return float(l['order_id'].nunique())
        if metric == 'qty':
            return float(l['qty_base'].sum())
        if metric == 'aov':
            n = l['order_id'].nunique()
            if n == 0:
                raise Abstain('no orders')
            return float(l['amount'].sum()) / n
    if metric == 'revenue':
        return float(o['rev'].sum())
    if metric == 'orders':
        return float(len(o))
    if metric == 'aov':
        if len(o) == 0:
            raise Abstain('no orders')
        return float(o['rev'].sum()) / len(o)
    if metric == 'customers':
        return float(o['cust_key'].nunique())
    raise Abstain('metric')


def detect_metric(q):
    n = q.n
    if q.has(r'trung binh|gia tri trung binh') and q.has(r'\bdon\b|moi don|1 don|mot don|tren don|lan mua'):
        return 'aov'
    u = unit_asked(q)
    if u and u not in ('tien', 'don') and not q.has(r'doanh thu|bao nhieu tien|het bao nhieu tien|tien ban|thu ve'):
        return 'qty'
    if q.has(r'doanh thu|tien ban|ban duoc (?:tong cong )?(?:bao nhieu|nhieu|het bao nhieu) tien|ban dc .*tien|thu ve'
             r'|ban ra .*tien|tien hang|thu duoc bao nhieu tien|ban .*duoc bao nhieu tien|mua tong cong bao nhieu tien'
             r'|mua .*bao nhieu tien|ban duoc bao nhieu tien|chot duoc bao nhieu tien|kiem duoc|tong tien ban'
             r'|ban .* het bao nhieu tien|ban duoc .*tien|tien thu ve|mang ve|thu (?:duoc|ve) .*tien|duoc bao nhieu tien'
             r'|duoc nhieu tien|het nhieu tien|\bban\b.*\bbao nhieu tien\b|tien ban .*bao nhieu'):
        return 'revenue'
    if q.has(r'\b(?:bao nhieu|may|so|dem|tong so|tong cong|nhieu)\s+(?:cai )?don\b(?! gia| vi)|\bso don\b|\bdon\b.*\bbao nhieu\b'):
        return 'orders'
    if unit_asked(q) or q.has(r'so luong'):
        return 'qty'
    return None


# ----------------------------------------------------------------------------
# handlers

SUBJECTIVE = (r'ngon nhat|than thien|lanh nghe|tay nghe|tan tam|le phep|chu dao|quy men|hay khen|khen'
              r'|dang tin|uy tin|de thuong|vui ve|dep trai|xinh|nhiet tinh|cham soc khach tot|phuc vu tot|kho chiu'
              r'|nen uong|lieu dung|uong may vien|tac dung phu|chong chi dinh|co tot khong|chat luong tot|hai long'
              r'|thich nhat|yeu thich nhat|ghet|tinh cach|dao duc|trung thuc|cham chi nhat(?! .*(?:ngay|so))|gioi nhat(?! .*doanh)'
              r'|thong minh|nang dong nhat|ca tinh|de chiu|kho khan nhat')


def _raw(q):
    import unicodedata
    return unicodedata.normalize('NFC', q.raw.lower())


def h_subjective(q):
    raw = _raw(q)
    if q.has(r'ba bau|co thai|cho con bu|tre (?:so sinh|em) (?:dung|uong)|(?:dung|uong|xai) (?:duoc|dc) khong|co (?:nen|duoc) (?:dung|uong)|lieu luong|cach dung|tac dung'):
        return {'type': 'unanswerable', 'reason': 'medical advice'}
    if re.search(r'khó tính', raw) or (q.has(r'kho tinh') and not q.has(r'trong kho|kho tinh (?:gia|theo|bang)')
                                         and q.has(r'khach|\bai\b')):
        return {'type': 'unanswerable', 'reason': 'subjective'}
    if q.has(SUBJECTIVE):
        # 'cham chi nhat' / 'gioi nhat' are defined by the rulebook
        if q.has(r'ban gioi nhat|cham chi nhat'):
            return None
        return {'type': 'unanswerable', 'reason': 'subjective'}
    if q.has(r'\bkhach (?:nu|nam)\b|\bgioi tinh\b|\bphai nu\b|\bkhach (?:la )?dan ong\b|\bphu nu\b|\bnam gioi\b|\bnu gioi\b'):
        if _has_gender(q.biz):
            return None
        return {'type': 'unanswerable', 'reason': 'no gender'}
    if (q.has(r'\bduong\s+[a-z]+(?:\s+[a-z]+){0,3}\b') and q.has(r'khach|o duong|nha o')) or \
            re.search(r'khách (?:ở|tại|nhà ở|sống ở) (?!đây|tiệm|cửa hàng|shop|quán)[A-ZĐÀ-Ỹ]', q.raw):
        if not _has_address(q.biz):
            return {'type': 'unanswerable', 'reason': 'no address'}
    if re.search(r'(?:huỷ|hủy|trả|trả lại|đổi|trả món|trả hàng)\s+(?:\w+\s+){0,2}(?:vì|do|bởi)\s', raw) or \
            q.has(r'ly do huy|ly do tra|nguyen nhan|vi giao tre|vi loi|vi goi nham|vi mua nham|bi loi nhieu hon|vi sao|tai sao|do dau|chu yeu vi'):
        return {'type': 'unanswerable', 'reason': 'no reason recorded'}
    if q.has(r'bao nhieu tuoi|do tuoi|tuoi cua khach|sinh nhat|ngay sinh|nghe nghiep|thu nhap'):
        return {'type': 'unanswerable', 'reason': 'not recorded'}
    return None


def _has_gender(biz):
    c = biz.customers
    if c is not None and 'gender' in c and c['gender'].notna().any():
        return True
    return False


def _has_address(biz):
    c = biz.customers
    if c is not None and 'address' in c and c['address'].notna().any():
        return True
    o = biz.orders
    return 'address' in o and o['address'].notna().any()


def h_cancel(q):
    if not q.has(r'\bhuy\b'):
        return None
    if q.has(r'khach .*huy nhieu nhat|hay doi y|hay huy'):
        return None
    p = need_period(q)
    o = orders_in(q, p, valid=False)
    f = parse_filters(q)
    prep_orders(q.biz)
    o = apply_order_filters(q, o, f)
    nc = int((o['status'] == 'cancelled').sum())
    if q.has(r'ti le|pct|chiem|bao nhieu phan'):
        if len(o) == 0:
            raise Abstain('no orders')
        return A('percent', share(nc, len(o)))
    if q.has(r'\bdon\b|bao nhieu|may'):
        return A('number', nc)
    return None


def _cmp_period(q, p):
    if p.kind == 'mtd' and not q.yoy:
        return P.previous(p, q.today)
    if q.yoy:
        try:
            return P.Period(p.start.replace(year=p.start.year - 1), p.end.replace(year=p.end.year - 1), p.kind)
        except ValueError:
            return P.Period(p.start.replace(year=p.start.year - 1, day=28), p.end.replace(year=p.end.year - 1, day=28), p.kind)
    if len(q.periods) >= 2:
        return q.periods[1]
    return P.previous(p, q.today)


CHANGE = (r'tang hay giam|giam hay tang|hon hay kem|kem hay hon|thay doi|len hay xuong|xuong hay len|cao hay thap'
          r'|thap hay cao|tang giam|chenh lech|\bchenh\b|tang bao nhieu|giam bao nhieu|nhieu hay it|it hay nhieu'
          r'|bien dong|so voi|so vs|so sanh|tang truong|tang them|giam di|sut giam|hon kem|cao thap|nhieu it')


def h_change(q):
    if not q.has(CHANGE):
        return None
    if q.has(r'\b(?:thang|tuan|quy|nam|ky|ben|kenh|ca|mat hang|loai) nao\b') or len(q.periods) > 2:
        return None
    if q.has(r'\bca chieu\b|\bca sang\b|\btang gia\b|\blen gia\b|\bdoi gia\b|\btet\b|\bgia moi\b|\bgia cu\b'):
        return None
    metric = detect_metric(q)
    if metric is None:
        return None
    p = need_period(q)
    if not (q.cmp_prev or q.yoy or len(q.periods) >= 2):
        return None
    p2 = _cmp_period(q, p)
    f = parse_filters(q)
    check_caps(q, f)
    v1 = metric_value(q, p, metric, f)
    v2 = metric_value(q, p2, metric, f)
    if q.has(r'pct|ti le'):
        return A('percent', pct(v1, v2))
    if metric in ('revenue', 'aov'):
        return A('number', rnd(v1) - rnd(v2) if metric == 'aov' else rnd(v1 - v2))
    return A('number', num(v1 - v2))


def h_tet(q):
    if not q.has(r'\btet\b'):
        return None
    tet = [p for p in q.periods if p.kind == 'tet']
    if not tet:
        return None
    if superlative(q) or topk(q):
        return None
    p = tet[0]
    if len(tet) >= 2 and q.has(CHANGE):
        metric = detect_metric(q) or 'revenue'
        f = parse_filters(q)
        v1, v2 = metric_value(q, tet[0], metric, f), metric_value(q, tet[1], metric, f)
        if q.has(r'pct|ti le'):
            return A('percent', pct(v1, v2))
        return A('number', rnd(v1 - v2) if metric in ('revenue', 'aov') else num(v1 - v2))
    metric = detect_metric(q) or 'revenue'
    f = parse_filters(q)
    check_caps(q, f)
    v = metric_value(q, p, metric, f)
    if q.has(r'truoc tet|ngay truoc'):
        m = re.search(r'(\d+) ngay (?:ngay )?(?:lien )?truoc', q.n)
        n = int(m.group(1)) if m else p.days()
        pb = P.Period(p.start - dt.timedelta(days=n), p.start - DAY, 'range')
        vb = metric_value(q, pb, metric, f)
        if q.has(r'pct|ti le'):
            return A('percent', pct(v, vb))
        return A('number', rnd(v - vb) if metric in ('revenue', 'aov') else num(v - vb))
    return A('number', rnd(v) if metric in ('revenue', 'aov') else num(v))


def h_shift_compare(q):
    if not (q.has(r'ca chieu') and q.has(r'ca sang')):
        return None
    if not cap_time(q.biz):
        raise Unanswerable('no time of day')
    metric = 'orders' if q.has(r'so don|\bdon\b') and not q.has(r'doanh thu|tien') else 'revenue'
    if q.has(r'trung binh'):
        metric = 'aov'
    p = need_period(q)
    f = parse_filters(q)
    f.shifts = []
    fa, fb = parse_filters(q), parse_filters(q)
    fa.shifts, fb.shifts = ['chieu'], ['sang']
    va = metric_value(q, p, metric, fa)
    vb = metric_value(q, p, metric, fb)
    if q.has(r'pct|ti le|lan luot|moi ca|tung ca') and not q.has(r'lan luot|moi ca|tung ca'):
        return A('percent', pct(va, vb))
    if q.has(r'lan luot|moi ca|tung ca'):
        order = [m for m in shift_mentions(q)]
        vals = {'chieu': va, 'sang': vb}
        return A('list', [num(rnd(vals[s])) for s in order])
    if metric == 'revenue':
        return A('number', rnd(va - vb))
    return A('number', num(va - vb))


def h_maxday(q):
    if not q.has(r'ngay nao|ngay mo|hom nao|bua nao|ngay gi|thu may|ngay thu may'):
        return None
    if not q.has(r'nhat'):
        return None
    if q.has(r'thu may'):
        return None
    p = need_period(q)
    o = orders_in(q, p)
    f = parse_filters(q)
    check_caps(q, f)
    prep_orders(q.biz)
    o = apply_order_filters(q, o, f)
    if q.has(r'dat khach|dong khach|nhieu don|so don|it khach|vang khach|nhieu khach') and not q.has(r'tien|doanh thu'):
        s = o.groupby('date').size()
    else:
        if f.line_level():
            l = apply_line_filters(q, goods_lines(lines_of(q, o)), f)
            l = l.merge(o[['order_id', 'date']], on='order_id')
            s = l.groupby('date')['amount'].sum()
        else:
            s = o.groupby('date')['rev'].sum()
    if len(s) == 0:
        raise Abstain('empty')
    if q.has(r'it nhat|thap nhat|e nhat|kem nhat|vang nhat|it tien nhat'):
        mn = s.min()
        d = s[s == mn].index.min()
    else:
        mx = s.max()
        d = s[s == mx].index.min()
    return A('date', d.strftime('%Y-%m-%d'))


def h_month_list(q):
    """Several periods: list of values, which one is max, monotonic boolean."""
    ps = q.periods
    mr = [p for p in ps if p.kind == 'monthrange']
    if mr:
        p = mr[0]
        months = []
        y, m = p.start.year, p.start.month
        while (y, m) <= (p.end.year, p.end.month):
            months.append(P.month_period(y, m))
            y, m = P.add_months(y, m, 1)
        ps = months
    if len(ps) < 2:
        return None
    metric = detect_metric(q) or ('revenue' if q.has(r'doanh thu') else None)
    if metric is None:
        return None
    if q.has(r'co phai|dung khong|phai khong|phai ko|dung vay khong|dung ko|dung chu|phai hong') and q.has(r'cao hon|thap hon|nhieu hon|it hon'):
        return _trend_claim(q, ps, metric)
    f = parse_filters(q)
    check_caps(q, f)
    vals = [metric_value(q, p, metric, f) for p in ps]
    if q.has(r'thang nao|ky nao|quy nao|tuan nao'):
        if q.has(r'thap nhat|it nhat|kem nhat'):
            i = int(np.argmin(vals))
        else:
            i = int(np.argmax(vals))
        p = ps[i]
        if p.kind == 'month':
            return A('date', f'{p.start.year}-{p.start.month:02d}')
        return None
    if q.has(r'luon|lien tuc|deu tang|deu giam|moi thang deu|thang sau .* hon thang truoc|co tang|co giam'):
        dec = q.has(r'giam|thap hon|it hon') and not q.has(r'tang|cao hon')
        ok = all((b < a) if dec else (b > a) for a, b in zip(vals, vals[1:]))
        return A('boolean', bool(ok))
    if q.has(r'lan luot|tung thang|theo thu tu|moi thang|tung ky|lan luot la|thu tu') or len(ps) >= 2:
        if metric in ('revenue', 'aov'):
            return A('list', [rnd(v) for v in vals])
        return A('list', [num(v) for v in vals])
    return None


def _trend_claim(q, ps, metric):
    """'co phai thang A cao hon thang B va thang C thap hon thang D': evaluate each claim."""
    f = parse_filters(q)
    s = q.n_nocmp
    claims = re.findall(r'(?:thang\s?|th\s?|t)?(\d{1,2}) (\d{4}|\d{2}) (?:\w+ ){0,3}?(cao hon|thap hon|nhieu hon|it hon) (?:thang\s?|th\s?|t)?(\d{1,2}) (\d{4}|\d{2})', s)
    if not claims:
        raise Abstain('trend claim not parsed')
    ok = True
    for m1, y1, rel, m2, y2 in claims:
        y1, y2 = P._year(y1), P._year(y2)
        a = metric_value(q, P.month_period(int(y1), int(m1)), metric, f)
        b = metric_value(q, P.month_period(int(y2), int(m2)), metric, f)
        ok &= (a > b) if rel in ('cao hon', 'nhieu hon') else (a < b)
    return A('boolean', bool(ok))


SHARE = r'trong (?:tong )?doanh thu|chiem|ti trong|ty trong|phan cua|bang bao nhieu pct|dong gop|chiem ti le|ti le .* tong|pct tong'


def h_share(q):
    if not q.has(SHARE):
        return None
    if not q.has(r'pct|ti trong|ty trong|ti le'):
        return None
    if q.has(r'khach .*tu \d+ don|it nhat \d+ don|\d+ don tro len|\bhuy\b'):
        return None
    p = need_period(q)
    f = parse_filters(q)
    check_caps(q, f)
    metric = 'revenue'
    if q.has(r'so don|\bdon\b') and not q.has(r'doanh thu|tien'):
        metric = 'orders'
    if q.has(r'luong|so (?:bich|goi|ly|lon|chai|hop)|tinh theo so'):
        metric = 'qty'
    if not f.any():
        return None
    base = Flt()
    if metric == 'qty':
        # share of a size/brand among the same unit family
        base.unit = f.unit
    if f.rx is not None and q.has(r'tong doanh thu|cua nha thuoc|toan bo'):
        pass
    if q.has(r'tong doanh thu may|doanh thu may') and q.has(r'phu kien'):
        pass
    v = metric_value(q, p, metric, f)
    tot = metric_value(q, p, metric, base)
    return A('percent', share(v, tot))


def h_multi(q):
    """Two or more values of one dimension: 'Zalo OA va tai quay ... lan luot / moi ben'."""
    f = parse_filters(q)
    metric = detect_metric(q)
    if metric is None:
        return None
    dims = []
    if len(f.channels) >= 2:
        dims = [('channels', c) for c in f.channels]
    elif len(f.pays) >= 2:
        dims = [('pays', c) for c in f.pays]
    elif len(f.shifts) >= 2:
        dims = [('shifts', c) for c in f.shifts]
    elif len(f.products) >= 2 and q.has(r'lan luot|moi mon|tung mon|moi loai|moi ben|tung loai|moi mat hang'):
        dims = [('products', c) for c in f.products]
    if not dims:
        return None
    check_caps(q, f)
    p = need_period(q)
    vals = []
    for dim, v in dims:
        g = parse_filters(q)
        setattr(g, dim, [v])
        vals.append(metric_value(q, p, metric, g))
    if q.has(r'ben nao|loai nao|kenh nao|ca nao|hinh thuc nao|cai nao'):
        return None
    if metric in ('revenue', 'aov'):
        return A('list', [rnd(v) for v in vals])
    return A('list', [num(v) for v in vals])


def h_threshold_orders(q):
    """Orders with value >= X / days with revenue >= X."""
    if not q.has(r'tro len|it nhat|tu \d|dat \d|tren \d|hon \d|qua \d|toi thieu|>='):
        return None
    if q.has(r'khach') or q.has(r'it nhat (?:mot|1) (?:mat hang|san pham|mon)|co (?:mua )?(?:mat hang|hang) nhom'):
        return None
    th = parse_money_threshold(q)
    if th is None:
        return None
    op, v = th
    p = need_period(q)
    o = orders_in(q, p)
    if q.has(r'may ngay|bao nhieu ngay|so ngay|may bua|bao nhieu bua|so bua|ngay nao|co nhieu ngay|nhieu bua'):
        s = o.groupby('date')['rev'].sum()
        return A('number', int(cmp(s, op, v).sum()))
    if q.has(r'\bdon\b'):
        return A('number', int(cmp(o['rev'], op, v).sum()))
    return None


def h_customers_n_orders(q):
    if not q.has(r'khach'):
        return None
    th = count_threshold(q, r'(?:don|lan)')
    if th is None:
        return None
    if not cap_customers(q.biz):
        raise Unanswerable('no customers')
    op, v = th
    p = need_period(q)
    o = orders_in(q, p)
    o = o[o['cust_key'].notna()]
    c = o.groupby('cust_key').size()
    n = int(cmp(c, op, v).sum())
    if q.has(r'pct|ti le|chiem|bao nhieu phan'):
        return A('percent', share(n, len(c)))
    return A('number', n)


def h_products_n_orders(q):
    if not q.has(r'mat hang|san pham|\bmon\b|\bhang\b'):
        return None
    th = count_threshold(q, r'(?:don)(?: khac nhau)?')
    if th is None:
        return None
    if not q.has(r'co mat|xuat hien|duoc mua|nam trong|co trong|mua o|ban trong|o tu|gop mat|di kem|duoc goi|goi trong|duoc dat'):
        return None
    op, v = th
    p = need_period(q)
    o = orders_in(q, p)
    l = goods_lines(lines_of(q, o))
    c = l.groupby('prod_key')['order_id'].nunique()
    return A('number', int(cmp(c, op, v).sum()))


# ---- rankings ---------------------------------------------------------------

_NUMW = {'hai': 2, 'ba': 3, 'bon': 4, 'tu': 4, 'nam': 5, 'sau': 6, 'bay': 7, 'tam': 8, 'chin': 9, 'muoi': 10}


def topk(q):
    m = q.find(r'\btop (hai|ba|bon|nam|sau|bay|tam|chin|muoi)\b|\b(hai|ba|bon|nam|sau|bay|tam|chin|muoi) (?:mat hang|san pham|mon|khach)\b')
    if m:
        return _NUMW[m.group(1) or m.group(2)]
    m = q.find(r'\btop\s?(\d{1,2})\b|\b(\d{1,2}) (?:best|mon|mat hang)? ?best ?selle')
    if m:
        return int(m.group(1) or m.group(2))
    m = q.find(r'(?<!thang )(?<!th )(?<!ngay )(?<!quy )(?<!nam )(?<!\b\d )(?<!\b\d\d )\b(\d{1,2})\s+(?:mat hang|san pham|mon|hang|sp|khach|khach hang|nguoi'
               r'|nhan vien|loai|thuoc|nha cung cap|ncc|mon an|do uong|dong may|dong xe|hang may|chiec may|may)\b')
    if m:
        return int(m.group(1))
    return None


def superlative(q):
    """'nhat' used as a superlative (not 'gan nhat' / 'it nhat N' / 'som nhat')."""
    s = re.sub(r'\bgan nhat\b|\bit nhat (?:\d+|mot|hai|ba)\b|\btoi thieu\b|\bsom nhat\b|\bduy nhat\b', ' ', q.n)
    return re.search(r'\bnhat\b|\btop\b|best ?selle|ban chay', s) is not None


def rank_order(s, keys_code, asc=False):
    """Sort a Series by value (desc) then code asc."""
    df = pd.DataFrame({'v': s.values, 'k': s.index, 'code': [keys_code(k) for k in s.index]})
    df = df.sort_values(['v', 'code'], ascending=[asc, True])
    return list(df['k']), list(df['v'])


def prod_code(biz):
    p = biz.products
    ids = dict(zip(p['prod_key'], p['prod_id'])) if p is not None and 'prod_id' in p else {}

    def f(k):
        c = ids.get(k)
        if c is None or (isinstance(c, float) and np.isnan(c)):
            return str(biz.prod_names.get(k, k))
        return str(c)
    return f


def h_product_rank(q):
    if not q.has(r'mat hang|san pham|\bmon\b|\bhang\b|\bthuoc\b|\bmon\b|\bloai\b|\bmay\b|\bsua\b|\bbim\b|\bbia\b|\bnuoc\b|do uong|\bmon\b'):
        return None
    if not (superlative(q) or topk(q)):
        return None
    prod_q = q.has(r'(?<!khach )\b(?:hang|mon|mat hang|san pham|thuoc|loai) nao\b')
    if q.has(r'nhan vien|nha cung cap|ncc\b|ngay nao|thang nao|\btho\b|gio nao|khung gio') or \
            (q.has(r'khach|\bai\b|nguoi mua') and not prod_q):
        return None
    p = need_period(q)
    o = orders_in(q, p)
    f = parse_filters(q)
    check_caps(q, f)
    prep_orders(q.biz)
    o = apply_order_filters(q, o, f)
    l = goods_lines(lines_of(q, o))
    pf = Flt()
    pf.category, pf.rx, pf.size = f.category, f.rx, f.size
    l = apply_line_filters(q, l, pf)
    asc = bool(q.has(r'cham nhat|it nhat|e nhat|thap nhat|kem nhat'))
    if q.has(r'doanh thu|tien|thu ve'):
        s = l.groupby('prod_key')['amount'].sum()
    elif q.has(r'loi nhat|lai nhat|loi nhieu nhat|lai nhieu nhat|lai gop'):
        return None
    elif q.has(r'ua chuong|nhieu don|co mat|xuat hien|nhieu nguoi mua'):
        s = l.groupby('prod_key')['order_id'].nunique()
    else:
        u = unit_asked(q)
        bu = q.biz.extra.get('base_unit', {})
        if u and q.has(r'nhieu ' + re.escape(u) + r' nhat') and any(v == u for v in bu.values()):
            l = l[l['prod_key'].map(lambda k: bu.get(k) == u)]
        s = l.groupby('prod_key')['qty_base'].sum()
    if len(s) == 0:
        raise Abstain('no sales')
    keys, vals = rank_order(s, prod_code(q.biz), asc=asc)
    k = topk(q)
    names = [q.biz.prod_names.get(x, x) for x in keys]
    if k:
        return A('list', names[:k])
    return A('string', names[0])


_PHONE = re.compile(r'[\s\-–(]*\(?\s*(?:\+?84|0)?\d[\d .\-]{7,}\d\s*\)?\s*$')


def clean_person(name):
    if not isinstance(name, str):
        return name
    s = _PHONE.sub('', name).strip(' -–()')
    return s or name


def cust_name_of(biz):
    """cust_key -> display name (phone numbers stripped)."""
    cache = getattr(biz, '_cnames', None)
    if cache is not None:
        return cache
    o = biz.orders
    names = {}
    c = biz.customers
    if c is not None and 'cust_id' in c and c['cust_id'].notna().any() and biz.has_cust_id:
        names.update({k: clean_person(v) for k, v in zip(c['cust_id'], c['name']) if isinstance(v, str)})
    sub = o[o['cust_key'].notna() & o['cust_name'].notna()]
    for k, g in sub.groupby('cust_key')['cust_name']:
        if k not in names:
            cl = g.map(clean_person)
            names[k] = cl.value_counts().index[0]
    biz._cnames = names
    return names


def customer_mentioned(q):
    """cust_key of a customer named in the question."""
    from loader import cust_key as ck_of
    names = cust_name_of(q.biz)
    best = None
    for k, nm in names.items():
        key = ck_of(nm)
        if key and len(key) >= 5 and re.search(r'(?<![a-z])' + re.escape(key) + r'(?![a-z])', q.n0):
            if best is None or len(key) > best[1]:
                best = (k, len(key))
    return best[0] if best else None


def h_customer_rank(q):
    if not q.has(r'khach|\bai\b.*\bmua\b|nguoi mua'):
        return None
    if q.has(r'(?<!khach )\b(?:hang|mon|mat hang|san pham|thuoc|loai) nao\b|\b(?:mat hang|san pham|mon)\b.*(?:ua chuong|ban chay)'):
        return None
    if not (superlative(q) or topk(q)):
        return None
    if q.has(r'\bno\b|con thieu|chua tra'):
        return None
    if not cap_customers(q.biz):
        raise Unanswerable('no customers')
    p = need_period(q)
    if q.has(r'doi y|huy nhieu|hay huy|huy don nhieu'):
        o = orders_in(q, p, valid=False)
        o = o[o['cust_key'].notna() & (o['status'] == 'cancelled')]
        s = o.groupby('cust_key').size()
    else:
        o = orders_in(q, p)
        o = o[o['cust_key'].notna()]
        if q.has(r'trung thanh|nhieu ngay|thuong xuyen|nhieu lan nhat|hay ghe|ghe nhieu'):
            if q.has(r'nhieu lan nhat') and q.has(r'so don'):
                s = o.groupby('cust_key').size()
            else:
                s = o.groupby('cust_key')['date'].nunique()
        elif q.has(r'sop|chiu chi|mo hau bao'):
            c = o.groupby('cust_key').agg(n=('order_id', 'size'), r=('rev', 'sum'))
            c = c[c['n'] >= 3]
            s = (c['r'] / c['n'])
        elif q.has(r'nhieu don|so don|nhieu lan|mua nhieu lan|hay mua'):
            s = o.groupby('cust_key').size()
        else:
            s = o.groupby('cust_key')['rev'].sum()
    if len(s) == 0:
        raise Abstain('no customers')
    biz = q.biz
    names = cust_name_of(biz)
    keys, vals = rank_order(s, lambda k: str(k))
    k = topk(q)
    out = [names.get(x, x) for x in keys]
    if k:
        return A('list', out[:k])
    return A('string', out[0])


def h_staff(q):
    if not q.has(r'nhan vien|\bnv\b|nguoi ban|\bnv\s?\d{2,4}\b'):
        return None
    o = q.biz.orders
    if 'staff' not in o or o['staff'].isna().all():
        raise Unanswerable('no staff data')
    p = need_period(q)
    if q.has(r'nhat') and q.has(r'nhan vien nao|nv nao|\bai\b|nguoi nao'):
        if q.has(r'cham chi|nhieu ngay'):
            oo = orders_in(q, p, valid=False)
            s = oo.groupby('staff')['date'].nunique()
        elif q.has(r'nhieu don|so don'):
            s = orders_in(q, p).groupby('staff').size()
        else:
            s = orders_in(q, p).groupby('staff')['rev'].sum()
        keys, vals = rank_order(s, lambda k: str(k))
        names = q.biz.extra.get('staff', {})
        return A('string', names.get(keys[0], keys[0]) if names else keys[0])
    f = parse_filters(q)
    if f.staff is None:
        return None
    metric = detect_metric(q) or 'revenue'
    return A('number', rnd(metric_value(q, p, metric, f)) if metric in ('revenue', 'aov') else num(metric_value(q, p, metric, f)))


_SLOT_STOP = {'hang', 'duoc', 'ra', 'cho', 'tai', 'qua', 'o', 'online', 'tong', 'bao', 'nhieu', 'tat', 'ca', 'het',
              'cac', 'moi', 'nhung', 'tien', 'hom', 'thang', 'tuan', 'nam', 'quy', 'ngay', 'la', 'cua', 'trong', 'dc',
              'mat', 'san', 'pham', 'mon', 'don', 'nhom', 'khach', 'size', 'ly', 'theo', 've', 'thuoc', 'shop', 'quan',
              'tiem', 'cua', 'hang', 'minh', 'mik', 'tinh', 'tong', 'cong', 'roi', 'nay', 'truoc', 'nhanh', 'it', 'nhat'}


def product_slot(q):
    """True when the question names an item that matched nothing ('ban <item> duoc bao nhieu')."""
    n = q.n0
    pats = [r'\b(?:mat hang|san pham)\s+((?:[a-z0-9]+ ){2,8}?)(?:trong|duoc|thang|tuan|quy|nam|hom|ngay|la|ban|thu|tu|o)\b',
            r'\bban\s+((?:[a-z0-9]+ ){2,8}?)duoc\b(?:\s+(?:tong cong\s+)?bao nhieu|\s+nhieu|\s+may)',
            r'\b(?:doanh thu|tien ban)(?: cua)?\s+((?:[a-z0-9]+ ){2,8}?)(?:trong|thang|tuan|quy|nam|hom|ngay|la|tu)\b']
    raw = q.raw
    for p in pats:
        m = re.search(p, n)
        if m:
            words = m.group(1).split()
            content = [w for w in words if w not in _SLOT_STOP and not re.fullmatch(r'\d+', w)]
            if len(content) < 2:
                continue
            # the slot must look like a product name: capitalised word or a size token
            first = content[0]
            hit = re.search(r'\b\d+(?:[.,]\d+)?\s?(?:g|kg|ml|l|lit|mm|cm|m|w|gb|mah|v|vien|goi|mieng)\b', m.group(1))
            from textnorm import norm as _n
            for w in re.findall(r'\w+', raw)[1:]:
                if re.fullmatch(r'(?:T|Th|Q|Quý|Tháng)\d+', w):
                    continue
                if w[:1].isupper() and _n(w) in content:
                    hit = True
                    break
            if hit:
                return True
    return False


def h_metric(q):
    metric = detect_metric(q)
    if metric is None:
        return None
    if q.has(r'\bno\b|tra no|thu no|nhap hang|nhap vao|gia nhap|\bvon\b|lai gop|loi gop|lai rong|chi phi'):
        return None
    if q.has(r'pct|ti le|ti trong|tang hay giam|hon hay kem|thay doi|len hay xuong|cao hay thap'):
        return None
    if metric == 'revenue' and q.has(r'(?:so|bao nhieu|may|dem) (?:cai )?don\b|(?:so|bao nhieu|may) ngay|may bua|\bdem\b') \
            and q.has(r'tro len|tro xuong|\btu \d|it nhat|toi thieu|\btren \d|\bduoi \d'):
        return None
    p = need_period(q)
    f = parse_filters(q, metric)
    if not f.any() and product_slot(q):
        raise Abstain('product named but not found')
    if metric == 'qty':
        if not f.products and not f.category and not f.size:
            f.unit = unit_asked(q)
            if f.unit is None:
                return None
        elif f.category and not f.products:
            pass
    check_caps(q, f)
    if len(q.periods) > 1 and not q.has(r'lan luot|tung'):
        return None
    v = metric_value(q, p, metric, f)
    if metric in ('revenue', 'aov'):
        return A('number', rnd(v))
    return A('number', num(v))


HANDLERS = [h_subjective, h_cancel, h_tet, h_shift_compare, h_change, h_month_list, h_customers_n_orders,
            h_products_n_orders, h_threshold_orders, h_share, h_maxday, h_customer_rank, h_staff, h_product_rank,
            h_multi, h_metric]
