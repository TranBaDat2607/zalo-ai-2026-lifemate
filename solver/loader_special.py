"""Shop books that do not fit the generic export layouts.

* repair book (one row per vehicle visit: Xe / Làm gì / Thợ / Tiền)
* phone shop book (STT order number, abbreviations sheet 'Ghi tắt', IMEI,
  side books for debt, instalments, warranty, trade-ins)
* non-goods lines inside invoices (Voucher, Dùng n điểm, Phí ship)
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

from loader import _num, cust_key, is_retail, status_of
from tables import read_sheets, table_from_rows
from textnorm import norm
from values import clean_str, parse_datetime, parse_number, parse_qty


def status_from_note(s):
    n = norm(s)
    if not n:
        return 'completed'
    if re.search(r'\bhuy\b|cancel|bom hang|khach bo|khong lay', n):
        return 'cancelled'
    if re.search(r'tra lai|tra hang|mang tra|doi tra|hoan hang|hoan tra|tra ve|khach tra', n):
        return 'returned'
    if re.search(r'dang giao|dang ship|dang van chuyen|chua giao|cho giao|dang gui', n):
        return 'shipping'
    return 'completed'


def debt_from_note(note, payable):
    """Outstanding debt written in an order note ('còn thiếu 73 nghìn', 'nợ 69k, khách ck đủ rồi')."""
    if not isinstance(note, str):
        return 0.0
    s = note.lower()
    if re.search(r'đủ|trả hết|xong|hết rồi|thanh toán đủ', s):
        return 0.0
    if re.search(r'nợ cả đơn|nợ toàn bộ|chưa trả đồng nào', s):
        return float(payable or 0)
    m = re.search(r'(?:còn thiếu|thiếu lại|thiếu|ghi nợ|ghi sổ|nợ)\s*([\d.,]+\s*(?:tr\d*|k|nghìn|ngàn|đ|đồng)?)', s)
    if m:
        txt = m.group(1).replace('nghìn', 'k').replace('ngàn', 'k').replace(' ', '')
        v = parse_number(txt)
        if v is not None:
            return float(v)
    m = re.search(r'(?:đưa trước|trả trước|ck|trả)\s*([\d.,]+\s*(?:tr\d*|k|nghìn|ngàn|đ|đồng)?).*còn lại', s)
    if m:
        v = parse_number(m.group(1).replace('nghìn', 'k').replace('ngàn', 'k').replace(' ', ''))
        if v is not None and payable:
            return max(float(payable) - v, 0.0)
    if re.search(r'nợ|ghi sổ|chưa trả', s):
        return float(payable or 0)
    return 0.0


def abbrev_map(tables):
    """{norm(abbrev or full name): (full name, type)} from a 'Ghi tắt' sheet."""
    out = {}
    for t in tables:
        h = [norm(c) for c in t.header]
        if 'ghi tat' not in h:
            continue
        ja = h.index('ghi tat')
        jn = next((j for j, c in enumerate(h) if c in ('ten hang', 'ten day du', 'hang')), None)
        jt = next((j for j, c in enumerate(h) if c in ('loai', 'loai hang', 'nhom')), None)
        for r in t.data:
            ab, full = clean_str(r[ja]), clean_str(r[jn]) if jn is not None else None
            typ = clean_str(r[jt]) if jt is not None else None
            if ab and full:
                out[norm(ab)] = (full, typ)
                out.setdefault(norm(full), (full, typ))
    return out


NONGOODS = re.compile(r'^(vouche?r?|dung\s+(\d+)\s+diem|phi ship|phi giao hang|phi van chuyen)\b')


def nongoods_kind(name):
    m = NONGOODS.match(norm(name))
    if not m:
        return None
    if m.group(1).startswith('vouche'):
        return 'voucher'
    if m.group(1).startswith('dung'):
        return 'points'
    return 'ship'


# ----------------------------------------------------------------------------
# repair book

def is_repair(tables):
    for t in tables:
        h = {norm(c) for c in t.header}
        if {'xe', 'lam gi'} <= h or ('lam gi' in h and 'tho' in h):
            return True
    return False


_PLATE = re.compile(r'(\d{2}[A-Z]{1,2}\d?-?\s?\d{3}\.?\d{2})', re.I)


def split_vehicle(s):
    s = clean_str(s) or ''
    m = _PLATE.search(s)
    if not m:
        return s, None
    plate = m.group(1).upper().replace(' ', '')
    model = (s[:m.start()] + s[m.end():]).strip()
    return model, plate


def load_repair(biz, tables):
    techs = []
    rows = []
    seq = 0
    for t in tables:
        h = [norm(c) for c in t.header]
        if 'ho ten' in h and any('hoa hong' in c for c in h):
            jn = h.index('ho ten')
            jc = next(j for j, c in enumerate(h) if 'hoa hong' in c)
            jd = next((j for j, c in enumerate(h) if 'ngay' in c), None)
            for r in t.data:
                techs.append(dict(name=clean_str(r[jn]), rate=parse_number(r[jc]),
                                  start=parse_datetime(r[jd]) if jd is not None else None))
            continue
        if 'lam gi' not in h:
            continue
        col = {c: j for j, c in enumerate(h)}
        for r in t.data:
            d = parse_datetime(r[col['ngay']], t.year) if 'ngay' in col else None
            if d is None:
                continue
            seq += 1
            model, plate = split_vehicle(r[col['xe']]) if 'xe' in col else (None, None)
            note = clean_str(r[col['ghi chu']]) if 'ghi chu' in col else None
            rows.append(dict(order_id=f'P{seq:06d}', date=d, rev=_num(r[col['tien']]) if 'tien' in col else np.nan,
                             staff_nick=clean_str(r[col['tho']]) if 'tho' in col else None,
                             vehicle=model, plate=plate, jobs=clean_str(r[col['lam gi']]), note=note,
                             warranty=bool(note and re.search(r'bao hanh', norm(note)))))
    O = pd.DataFrame(rows)
    O['rev'] = O['rev'].fillna(0.0)
    # nickname -> full name (given name = last word)
    nick2full = {}
    for tch in techs:
        if tch['name']:
            nick2full.setdefault(norm(tch['name']).split()[-1], tch['name'])

    def full(nick):
        n = norm(nick)
        n = re.sub(r'^(a|anh|chi|c|em|tho|ong|ba)\s+', '', n)
        n = n.split()[-1] if n else n
        return nick2full.get(n, nick)
    O['staff'] = O['staff_nick'].map(full)
    O['status'] = 'completed'
    O['dt'] = pd.NaT
    O['cust_name'] = None
    O['cust_id'] = None
    biz.orders = O
    biz.lines = None
    biz.products = pd.DataFrame(columns=['prod_key', 'prod_id', 'name', 'category', 'base_unit'])
    biz.prod_names, biz.prod_spellings = {}, {}
    biz.extra['base_unit'] = {}
    biz.extra['techs'] = techs
    biz.extra['kind'] = 'repair'
    biz.extra['staff'] = {t['name']: t['name'] for t in techs}


# ----------------------------------------------------------------------------
# phone shop books

def is_phone_book(tables):
    names = {norm(t.sheet or '') for t in tables}
    has_abbrev = any('ghi tat' in {norm(c) for c in t.header} for t in tables)
    has_imei = any('imei' in {norm(c) for c in t.header} for t in tables)
    return has_abbrev and has_imei


def load_phone(biz, tables):
    ab = abbrev_map(tables)
    lines = []
    for t in tables:
        h = [norm(c) for c in t.header]
        if 'stt' not in h or 'hang' not in h or 'imei' not in h:
            continue
        col = {c: j for j, c in enumerate(h)}
        prev = {}
        for r in t.data:
            g = lambda c: (r[col[c]] if c in col and r[col[c]] not in ('', None) else None)
            stt = clean_str(g('stt'))
            if stt is None:
                stt = prev.get('stt')
            else:
                if stt != prev.get('stt'):
                    prev = {'stt': stt}
            if stt is None:
                continue
            rec = {'order_id': stt}
            for c in ('ngay', 'khach', 'bot', 'thu may cu', 'khach tra', 'ghi chu'):
                v = g(c)
                if v is None:
                    v = prev.get(c)
                else:
                    prev[c] = v
                rec[c] = v
            raw = clean_str(g('hang'))
            if raw is None:
                continue
            full, typ = ab.get(norm(raw), (raw, None))
            q, _ = parse_qty(g('sl'))
            rec.update(prod_name=full, prod_raw=raw, ptype=typ, imei=clean_str(g('imei')),
                       qty=q if q is not None else 1.0, price=_num(g('gia')), amount=_num(g('thanh tien')),
                       year=t.year)
            lines.append(rec)
    L = pd.DataFrame(lines)
    L['amount'] = L['amount'].fillna(L['qty'] * L['price'])
    L['date_p'] = [parse_datetime(v, y) for v, y in zip(L['ngay'], L['year'])]
    L['prod_key'] = L['prod_name'].map(norm)
    first = L.groupby('order_id', sort=False).first()
    O = pd.DataFrame({'order_id': first.index})
    O['date'] = [d.date() if d else None for d in first['date_p']]
    O['cust_name'] = first['khach'].map(clean_str).values
    O['order_disc'] = first['bot'].map(_num).values
    O['tradein'] = first['thu may cu'].map(_num).values if 'thu may cu' in first else np.nan
    O['cust_paid_net'] = first['khach tra'].map(_num).values
    O['note'] = first['ghi chu'].map(clean_str).values
    O['status'] = O['note'].map(status_from_note)
    O['rev'] = O['order_id'].map(L.groupby('order_id')['amount'].sum()).values
    O['dt'] = pd.NaT
    O['cust_id'] = None
    biz.orders = O
    lines_df = pd.DataFrame({'order_id': L['order_id'], 'prod_key': L['prod_key'], 'prod_name': L['prod_name'],
                             'unit': '', 'qty': L['qty'], 'qty_base': L['qty'], 'price': L['price'],
                             'line_disc': np.nan, 'amount': L['amount'], 'cost': np.nan, 'line_no': None,
                             'category_line': L['ptype'], 'imei': L['imei'], 'is_fee': False})
    biz.lines = lines_df
    names = L.groupby('prod_key')['prod_name'].agg(lambda s: s.value_counts().index[0])
    typ = L.groupby('prod_key')['ptype'].agg(lambda s: s.dropna().value_counts().index[0] if s.notna().any() else None)
    biz.products = pd.DataFrame({'prod_key': names.index, 'prod_id': None, 'name': names.values,
                                 'category': typ.reindex(names.index).values, 'base_unit': None})
    biz.prod_names = dict(zip(names.index, names.values))
    sp = {}
    for k, raw, full in zip(L['prod_key'], L['prod_raw'], L['prod_name']):
        sp.setdefault(k, set()).update([raw, full])
    biz.prod_spellings = {k: list(v) for k, v in sp.items()}
    biz.extra['base_unit'] = {k: 'cai' for k in names.index}
    biz.units = {k: {'cai': 1.0, 'chiec': 1.0, 'may': 1.0} for k in names.index}
    biz.extra['kind'] = 'phone'
    _phone_side_books(biz, tables)


def _phone_side_books(biz, tables):
    for t in tables:
        h = [norm(c) for c in t.header]
        sh = norm(t.sheet or '') + ' ' + norm(t.fname)
        col = {c: j for j, c in enumerate(h)}
        if 'gia thu' in col:
            rows = []
            for r in t.data:
                rows.append(dict(date=parse_datetime(r[col['ngay']], t.year) if 'ngay' in col else None,
                                 cust=clean_str(r[col['khach']]) if 'khach' in col else None,
                                 model=clean_str(r[col.get('may thu', col.get('may', 0))]),
                                 imei=clean_str(r[col['imei']]) if 'imei' in col else None,
                                 price=_num(r[col['gia thu']]),
                                 order_id=clean_str(next((r[j] for c, j in col.items() if 'tru vao' in c), None))))
            biz.extra['tradein'] = pd.DataFrame(rows)
        elif 'so no' in sh or ('no' in col and 'da tra' in col):
            rows = []
            cur = None
            for r in t.data:
                g = lambda c: (r[col[c]] if c in col and r[col[c]] not in ('', None) else None)
                note = clean_str(g('ghi chu'))
                stt = None
                if note:
                    m = re.search(r'stt\s*(\d+)', norm(note))
                    if m:
                        stt = m.group(1)
                if g('no') is not None:
                    cur = dict(order_id=stt, date=parse_datetime(g('ngay'), t.year), cust=clean_str(g('khach')),
                               phone=clean_str(g('sdt')), debt=_num(g('no')))
                    rows.append(dict(kind='debt', **cur, amount=cur['debt'], pay_date=None))
                if g('da tra') is not None and cur is not None:
                    rows.append(dict(kind='pay', order_id=cur['order_id'], date=cur['date'], cust=cur['cust'],
                                     phone=cur['phone'], debt=None, amount=_num(g('da tra')),
                                     pay_date=parse_datetime(g('ngay tra'), t.year)))
            biz.extra['debt_book'] = pd.DataFrame(rows)
        elif 'tra gop' in sh or 'gop thang' in col:
            rows = []
            cur = None
            for r in t.data:
                g = lambda c: (r[col[c]] if c in col and r[col[c]] not in ('', None) else None)
                if g('gia') is not None or g('so ky') is not None:
                    cur = dict(order_id=clean_str(g('stt')), date=parse_datetime(g('ngay'), t.year),
                               cust=clean_str(g('khach')), phone=clean_str(g('sdt')), model=clean_str(g('may')),
                               total=_num(g('gia')), down=_num(g('tra truoc')), n=_num(g('so ky')),
                               per=_num(g('gop thang')), due_day=_num(g('han gop ngay')), pays=[])
                    rows.append(cur)
                if cur is not None and g('da gop') is not None:
                    cur['pays'].append((parse_datetime(g('ngay gop'), t.year), _num(g('da gop'))))
            biz.extra['installments'] = rows
        elif 'bao hanh' in sh or 'ngay nhan' in col:
            rows = []
            for r in t.data:
                g = lambda c: (r[col[c]] if c in col and r[col[c]] not in ('', None) else None)
                rows.append(dict(date=parse_datetime(g('ngay nhan'), t.year), cust=clean_str(g('khach')),
                                 model=clean_str(g('may')), imei=clean_str(g('imei')),
                                 buy_date=parse_datetime(g('ngay mua'), t.year), issue=clean_str(g('loi')),
                                 action=clean_str(g('xu ly'))))
            biz.extra['warranty'] = pd.DataFrame(rows)
