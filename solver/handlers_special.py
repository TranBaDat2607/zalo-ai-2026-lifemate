"""Handlers for repair shops, phone shops and mother & baby stores (rulebook §6)."""
import calendar
import datetime as dt
import re
from collections import Counter

import numpy as np
import pandas as pd

import periods as P
from handlers import (CHANGE, Flt, apply_line_filters, detect_metric, metric_value, need_period, parse_filters,
                      rank_order, superlative, topk, unit_asked)
from qa import A, Abstain, DAY, TS, Unanswerable, goods_lines, lines_of, num, orders_in, pct, rnd, share, window
from textnorm import norm

WEEKDAYS = {'thu hai': 0, 'thu 2': 0, 'thu ba': 1, 'thu 3': 1, 'thu tu': 2, 'thu 4': 2, 'thu nam': 3, 'thu 5': 3,
            'thu sau': 4, 'thu 6': 4, 'thu bay': 5, 'thu 7': 5, 'chu nhat': 6, 'cn': 6}


def kind(q):
    return q.biz.extra.get('kind')


# ----------------------------------------------------------------------------
# repair shop

def model_canon(biz):
    """raw vehicle model -> canonical model (case, spacing, abbreviations, numeric suffix)."""
    cache = biz.extra.get('_mcanon')
    if cache is not None:
        return cache
    o = biz.orders
    cnt = Counter(o['vehicle'].dropna())

    def base(s):
        n = norm(s)
        n = re.sub(r'\s+(?:\d{2,3}|fi|x|s|cc)$', '', n)
        return n
    forms = Counter()
    for s, c in cnt.items():
        forms[base(s)] += c
    # canonical forms: multi-letter forms; abbreviations map onto them
    canon_keys = sorted(forms, key=lambda k: (-len(k.split()), -len(k), -forms[k]))
    squash = {}
    for k in canon_keys:
        squash.setdefault(k.replace(' ', ''), k)
    out = {}
    for s in cnt:
        b = base(s)
        target = squash.get(b.replace(' ', ''), b)
        if len(b.split()) == 1:
            for k in canon_keys:
                ws = k.split()
                if k == b or len(ws) < 1:
                    continue
                if len(ws) >= 2 and (b == ''.join(w[0] for w in ws) or b in ws[1:] or (b == ws[0] and forms[b] == 0)):
                    target = k
                    break
                if len(ws) == 1 and len(b) >= 2 and ws[0].startswith(b) and ws[0] != b:
                    target = k
                    break
            if b == 'wave' and any(k.startswith('wave ') for k in canon_keys):
                target = max((k for k in canon_keys if k.startswith('wave ') and k != 'wave rsx'), key=lambda k: forms[k],
                             default=target)
        out[s] = target
    # pretty name per canonical key: most frequent raw spelling whose base equals the key
    pretty = {}
    for s, c in cnt.most_common():
        k = out[s]
        if k not in pretty and base(s) == k:
            pretty[k] = s
    for s, k in out.items():
        pretty.setdefault(k, s)
    biz.extra['_mcanon'] = (out, pretty)
    return out, pretty


JOBS = [
    ('sua dien', r'\b(?:sua dien|lam dien|dien)\b', r'den|bong|ac quy|binh'),
    ('nhot lap', r'\b(?:nhot lap|dau lap|lap|nhot hop so|dau hop so)\b', None),
    ('nhot may', r'\b(?:nhot|castrol|motul|shell)\b', r'lap|hop so'),
    ('va', r'\bva\b', None),
    ('thay ruot', r'\bruot\b', r'\bva\b'),
    ('rua xe', r'\brua\b', r'kim phun'),
    ('bao duong', r'\b(?:bao duong|bd|bdtq)\b', None),
    ('noi', r'\bnoi\b', None),
    ('bugi', r'\bbu ?gi\b', None),
    ('phanh', r'\b(?:bo thang|bo truoc|bo sau|ma phanh|bo)\b', None),
    ('loc gio', r'\bloc gio\b', None),
    ('sen', r'\b(?:sen|nhong|xich|nsd)\b', None),
    ('curoa', r'\bcu ?roa\b', None),
    ('vo xe', r'\bvo\b', None),
    ('kim phun', r'\bkim phun\b', None),
    ('nuoc mat', r'\bnuoc (?:lam )?mat\b', None),
    ('ac quy', r'\b(?:ac quy|binh)\b', None),
    ('den', r'\b(?:bong den|den pha|den)\b', None),
]


def job_set(jobs):
    out = set()
    for part in re.split(r'\s*[,+;/]\s*|\s+va\s+(?=[a-z])', norm(jobs or '')):
        part = part.strip()
        if not part:
            continue
        for name, rx, ex in JOBS:
            if re.search(rx, part) and not (ex and re.search(ex, part)):
                out.add(name)
    return out


JOB_Q = [('sua dien', r'sua dien|lam dien'), ('nhot lap', r'dau lap|nhot lap|\blap\b|nhot hop so|dau hop so'),
         ('va', r'\bva\b'), ('thay ruot', r'thay ruot|ruot moi'), ('nhot may', r'thay nhot|nhot may'),
         ('rua xe', r'rua xe'), ('bao duong', r'bao duong'), ('bugi', r'\bbu ?gi\b'), ('loc gio', r'loc gio'),
         ('curoa', r'cu ?roa'), ('vo xe', r'thay vo|vo xe'), ('kim phun', r'kim phun'), ('noi', r've sinh noi|lam noi'),
         ('ac quy', r'ac quy|binh dien'), ('den', r'bong den|den pha'), ('phanh', r'ma phanh|bo thang')]


def mechanic_named(q):
    for t in q.biz.extra.get('techs', []):
        nm = t['name']
        if nm and re.search(r'\b' + re.escape(norm(nm)) + r'\b', q.n0):
            return nm
    return None


def commissions(q, p):
    o = orders_in(q, p)
    rate = {t['name']: (t['rate'] or 0) / 100 for t in q.biz.extra.get('techs', [])}
    s = o.groupby('staff')['rev'].sum()
    return pd.Series({k: v * rate.get(k, 0) for k, v in s.items() if k in rate})


def h_repair(q):
    if kind(q) != 'repair':
        return None
    biz = q.biz
    if q.has(r'nhiet tinh|vui ve|than thien|tay nghe|gioi nghe|lanh nghe|cau tha|uy tin'):
        return {'type': 'unanswerable', 'reason': 'subjective'}
    p = q.periods[0] if q.periods else None
    if p is None:
        return None
    o = orders_in(q, p)
    paid = o[~o['warranty']]
    mech = mechanic_named(q)
    # commission
    if q.has(r'hoa hong|hue hong'):
        cm = commissions(q, p)
        if mech:
            return A('number', rnd(cm.get(mech, 0.0)))
        if q.has(r'\bbang\b|tung tho|moi tho|cac tho|moi nguoi|moi tho') or (not superlative(q) and not q.has(r'\bai\b|tho nao|tho mo')):
            cm = cm[cm > 0].sort_values(ascending=False)
            return A('table', {'columns': ['tho', 'hoa_hong'], 'rows': [[k, rnd(v)] for k, v in cm.items()]})
        keys, _ = rank_order(cm, lambda k: str(k))
        return A('string', keys[0])
    # per mechanic
    if mech or q.has(r'\btho\b'):
        if mech and not superlative(q):
            om = o[o['staff'] == mech]
            if q.has(r'tien|doanh thu|ban'):
                return A('number', rnd(om['rev'].sum()))
            return A('number', int(len(om)))
        if superlative(q) or q.has(r'tho nao|tho mo|\bai\b'):
            if q.has(r'cham chi|nhieu ngay'):
                s = o.groupby('staff')['date'].nunique()
            elif q.has(r'tien|doanh thu'):
                s = o.groupby('staff')['rev'].sum()
            else:
                s = o.groupby('staff').size()
            keys, _ = rank_order(s, lambda k: str(k))
            return A('string', keys[0])
    # average per paid ticket
    if q.has(r'trung binh|binh quan') and q.has(r'phieu|luot|\bxe\b|don') and not q.has(r'\bthu (?:hai|ba|tu|nam|sau|bay|\d)\b|moi ngay|1 ngay'):
        if len(paid) == 0:
            raise Abstain('no paid tickets')
        return A('number', rnd(paid['rev'].sum() / len(paid)))
    # average per weekday
    wd = None
    for w, i in WEEKDAYS.items():
        if re.search(r'\b' + w + r'\b', q.n):
            wd = i
            break
    if wd is not None and q.has(r'trung binh|binh quan'):
        a, b = window(q, p)
        ndays = sum(1 for k in range((b - a).days + 1) if (a + dt.timedelta(days=k)).weekday() == wd)
        n = int((o['date'].dt.weekday == wd).sum())
        if ndays == 0:
            raise Abstain('no such weekday')
        if q.has(r'tien|doanh thu'):
            return A('number', rnd(o.loc[o['date'].dt.weekday == wd, 'rev'].sum() / ndays))
        return A('number', rnd(n / ndays, 2))
    # warranty
    if q.has(r'bao hanh'):
        w = o[o['warranty']]
        if q.has(r'pct|ti le|chiem'):
            return A('percent', share(len(w), len(paid)))
        return A('number', int(len(w)))
    # vehicles
    if q.has(r'chiec|xe khac nhau|bien so|khach quen|tu \d+ lan|nhieu lan|may xe|bao nhieu xe') and not q.has(r'luot'):
        out, pretty = model_canon(biz)
        oo = o
        model = None
        for raw, k in sorted(out.items(), key=lambda x: -len(norm(x[1]))):
            if re.search(r'\b' + re.escape(k) + r'\b', q.n0) or re.search(r'\b' + re.escape(norm(raw)) + r'\b', q.n0):
                model = k
                break
        if model:
            oo = o[o['vehicle'].map(lambda s: out.get(s) == model)]
        if q.has(r'khach quen|tu \d+ lan|nhieu lan|it nhat \d+ lan|\d+ lan tro len'):
            m = re.search(r'(\d+) lan', q.n)
            k = int(m.group(1)) if m else 2
            c = oo[~oo['warranty']].groupby('plate').size()
            return A('number', int((c >= k).sum()))
        return A('number', int(oo['plate'].nunique()))
    if q.has(r'dong xe') and superlative(q):
        out, pretty = model_canon(biz)
        s = o.groupby(o['vehicle'].map(out))['rev'].sum() if q.has(r'tien|doanh thu') else o.groupby(o['vehicle'].map(out)).size()
        keys, _ = rank_order(s, lambda k: str(k))
        return A('string', pretty.get(keys[0], keys[0]))
    # jobs
    for name, rx in JOB_Q:
        if re.search(rx, q.n):
            js = paid['jobs'].map(job_set)
            return A('number', int(js.map(lambda s: name in s).sum()))
    # ticket counts (incl. warranty)
    if q.has(r'phieu|luot xe|nuot xe|luot|xe vao|xe vo|xe toi') and q.has(r'bao nhieu|may|dem|so') \
            and not q.has(r'tien|doanh thu'):
        return A('number', int(len(o)))
    return None


# ----------------------------------------------------------------------------
# phone shop

def phone_kind(q):
    return kind(q) == 'phone'


def tradein_by_order(biz):
    t = biz.extra.get('tradein')
    o = biz.orders
    if t is not None and len(t):
        s = t.groupby('order_id')['price'].sum()
        return o['order_id'].map(s).fillna(0.0)
    if 'tradein' in o:
        return o['tradein'].fillna(0.0)
    return pd.Series(0.0, index=o.index)


def is_phone_line(l):
    return l['category_line'].map(lambda c: norm(c) in ('may moi', 'may cu') if isinstance(c, str) else bool(False)) \
        | l['imei'].notna()


def model_line(name):
    """'iPhone 16 Pro 128GB Đen' -> 'iPhone 16 Pro'; names without ...GB stay whole."""
    if not isinstance(name, str):
        return name
    m = re.search(r'\s\d+\s?(?:GB|TB)\b', name, re.I)
    return name[:m.start()].strip() if m else name.strip()


def brand_of(name):
    if not isinstance(name, str) or not name.strip():
        return None
    w = name.split()[0]
    return 'Apple' if norm(w) in ('iphone', 'ipad', 'apple') else w


def phone_lines(q, o):
    l = lines_of(q, o)
    return l[is_phone_line(l)]


def installment_rows(q, d):
    """Instalment contracts placed by day d with paid / due / overdue amounts at end of d."""
    biz = q.biz
    o = biz.orders.set_index('order_id')
    out = []
    for r in biz.extra.get('installments', []) or []:
        if r['date'] is None or r['date'].date() > d:
            continue
        oid = r['order_id']
        if oid in o.index and o.loc[oid, 'status'] == 'cancelled':
            continue
        n = int(r['n'] or 0)
        per = float(r['per'] or 0.0)
        day = int(r['due_day'] or 1)
        y, m = r['date'].year, r['date'].month
        dues = []
        for i in range(1, n + 1):
            yy, mm = P.add_months(y, m, i)
            dues.append(dt.date(yy, mm, min(day, calendar.monthrange(yy, mm)[1])))
        paid = sum(a for t, a in r['pays'] if t is not None and a and t.date() <= d)
        total = float(r['total'] or 0) - float(r['down'] or 0)
        n_due_incl = sum(1 for x in dues if x <= d)
        n_due_before = sum(1 for x in dues if x < d)
        overdue_amt = max(per * n_due_before - paid, 0.0)
        out.append(dict(order_id=oid, cust=r['cust'], date=r['date'], remaining=max(total - paid, 0.0),
                        n_due=n_due_incl, overdue=overdue_amt, paid=paid))
    return out


def phone_debts(q, d):
    """Outstanding customer debt per order at end of d: debt book + instalments."""
    from loader import cust_key
    biz = q.biz
    rows = []
    db = biz.extra.get('debt_book')
    o = biz.orders.set_index('order_id')
    if db is not None and len(db):
        heads = db[db['kind'] == 'debt']
        pays = db[db['kind'] == 'pay']
        for _, h in heads.iterrows():
            if h['date'] is None or pd.isna(h['date']) or h['date'].date() > d:
                continue
            oid = h['order_id']
            if oid in o.index and o.loc[oid, 'status'] == 'cancelled':
                continue
            pp = pays[(pays['order_id'] == oid) & (pays['cust'] == h['cust'])]
            paid = sum(a for t, a in zip(pp['pay_date'], pp['amount'])
                       if t is not None and not pd.isna(t) and a and t.date() <= d)
            rows.append(dict(order_id=oid, cust=h['cust'], key=cust_key(h['cust']), date=h['date'],
                             owed=max(float(h['debt'] or 0) - paid, 0.0)))
    for r in installment_rows(q, d):
        rows.append(dict(order_id=r['order_id'], cust=r['cust'], key=cust_key(r['cust']), date=r['date'],
                         owed=r['remaining']))
    return [r for r in rows if r['owed'] > 0.5]


def h_phone(q):
    if not phone_kind(q):
        return None
    biz = q.biz
    from loader import cust_key
    p = q.periods[0] if q.periods else None
    day_p = [x for x in q.periods if x.kind == 'day']
    d_asof = min(day_p[0].start, q.today) if day_p else q.today

    def named_rows(rows):
        for r in rows:
            k = cust_key(r['cust'])
            if k and len(k) >= 5 and re.search(r'\b' + re.escape(k) + r'\b', q.n0):
                return [x for x in rows if cust_key(x['cust']) == k]
        return None

    # ---- instalments ----
    if q.has(r'tra gop|tien gop|ky gop|han gop|da gop|chua gop|cham gop|gop (?:thang|ky|tien|toi|den|du)|con gop|nguoi gop|khach gop|hop dong'):
        st = installment_rows(q, d_asof)
        if not st:
            raise Abstain('no instalments')
        sel = named_rows(st)
        rows = sel if sel is not None else st
        money = q.has(r'bao nhieu tien|tien|so tien|bao nhieu\b(?! ky| nguoi| khach)')
        if q.has(r'qua han|tre han|cham gop|cham dong|chua dong .*qua han'):
            if q.has(r'may nguoi|bao nhieu nguoi|bao nhieu khach|may khach|so khach|\bnguoi\b'):
                return A('number', len({cust_key(r['cust']) for r in rows if r['overdue'] > 0.5}))
            return A('number', rnd(sum(r['overdue'] for r in rows)))
        if q.has(r'chua gop|chua tra|con thieu|con no|con lai|chua dong'):
            return A('number', rnd(sum(r['remaining'] for r in rows)))
        if q.has(r'toi han|den han') and q.has(r'may ky|bao nhieu ky|so ky'):
            return A('number', int(sum(r['n_due'] for r in rows)))
        return None
    # ---- debt ----
    if q.has(r'\bno\b|cong no|con thieu') and not q.has(r'tra no|thu no|tra bot'):
        rows = phone_debts(q, d_asof)
        if q.has(r'tuoi|qua \d+ ngay|tren \d+ ngay'):
            rng = re.findall(r'(\d+)\s*[–\-]\s*(\d+)', q.raw.lower())
            age = [(TS(d_asof) - r['date']).days for r in rows]
            if rng:
                out = [rnd(sum(r['owed'] for r, g in zip(rows, age) if int(a) <= g <= int(b))) for a, b in rng]
                mm = re.search(r'(?:tren|qua|hon) (\d+)', q.n)
                if mm:
                    out.append(rnd(sum(r['owed'] for r, g in zip(rows, age) if g > int(mm.group(1)))))
                return A('list', out)
            mm = re.search(r'(?:qua|tren|hon) (\d+) ngay', q.n)
            if mm:
                return A('number', rnd(sum(r['owed'] for r, g in zip(rows, age) if g > int(mm.group(1)))))
        sel = named_rows(rows)
        if sel is not None:
            return A('number', rnd(sum(r['owed'] for r in sel)))
        if q.has(r'may khach|bao nhieu khach|so khach|may nguoi|bao nhieu nguoi'):
            return A('number', len({r['key'] for r in rows if r['key']}))
        if superlative(q) or topk(q):
            s = pd.Series({}, dtype=float)
            for r in rows:
                if r['key']:
                    s[r['key']] = s.get(r['key'], 0.0) + r['owed']
            names = {r['key']: r['cust'] for r in rows}
            keys, vals = rank_order(s, lambda k: str(k))
            k = topk(q) or 1
            from handlers import clean_person
            tab = [[clean_person(names[x]), rnd(v)] for x, v in zip(keys[:k], vals[:k])]
            if q.has(r'moi nguoi|tung nguoi|so no|bao nhieu'):
                return A('table', {'columns': ['khach_hang', 'so_tien_no'], 'rows': tab})
            return A('string', tab[0][0]) if k == 1 else A('list', [t[0] for t in tab])
        return A('number', rnd(sum(r['owed'] for r in rows)))
    if p is None:
        return None
    o = orders_in(q, p)
    # ---- amount payable ----
    if q.has(r'phai tra|khach tra'):
        if q.has(r'trung binh|moi don|binh quan'):
            if len(o) == 0:
                raise Abstain('no orders')
            return A('number', rnd(o['cust_paid_net'].sum() / len(o)))
        return A('number', rnd(o['cust_paid_net'].sum()))
    # ---- trade-ins ----
    if q.has(r'may cu') and q.has(r'\bthu\b|doi|tru vao'):
        ti = tradein_by_order(biz).reindex(o.index).fillna(0.0)
        if q.has(r'pct|ti le|chiem'):
            return A('percent', share(int((ti > 0).sum()), len(o)))
        if q.has(r'may don|bao nhieu don|so don'):
            return A('number', int((ti > 0).sum()))
        return A('number', rnd(ti.sum()))
    # ---- warranty ----
    if q.has(r'bao hanh'):
        w = biz.extra.get('warranty')
        if w is None:
            raise Unanswerable('no warranty data')
        a, b = window(q, p)
        ww = w[(w['date'] >= TS(a)) & (w['date'] < TS(b) + pd.Timedelta(days=1))]
        if q.has(r'dong may nao|may nao|dong nao') and superlative(q):
            s = ww.groupby(ww['model'].map(lambda m: norm(m)))['date'].size()
            keys, _ = rank_order(s, lambda k: str(k))
            pretty = {norm(m): m for m in ww['model'].dropna()}
            return A('string', pretty.get(keys[0], keys[0]))
        if q.has(r'pct|ti le|chiem'):
            phones = phone_lines(q, o)['qty'].sum()
            return A('percent', share(len(ww), phones))
        if q.has(r'may ban ra|ban ra thang|da mang lai|mang lai bao hanh|mang toi bao hanh|bao nhieu may|may ban'):
            l = phone_lines(q, o)
            imeis = set()
            for x in l['imei'].dropna():
                imeis.update(i.strip() for i in str(x).split('/'))
            w2 = w[w['date'] < TS(q.today) + pd.Timedelta(days=1)]
            return A('number', int(len(imeis & set(w2['imei'].dropna().astype(str)))))
        return A('number', int(len(ww)))
    # ---- phones sold ----
    l = lines_of(q, o)
    cat = l['category_line'].map(lambda c: norm(c) if isinstance(c, str) else '')
    phones = l[cat.isin(['may moi', 'may cu']) | l['imei'].notna()]
    if q.has(r'dong may') and (superlative(q) or topk(q)):
        s = phones.groupby(phones['prod_name'].map(model_line))['qty'].sum()
        keys, _ = rank_order(s, lambda k: str(k))
        k = topk(q)
        return A('list', keys[:k]) if k else A('string', keys[0])
    sub = phones
    if q.has(r'may cu|dien thoai cu|hang cu'):
        sub = l[cat == 'may cu']
    elif q.has(r'may moi|dien thoai moi|hang moi'):
        sub = l[cat == 'may moi']
    brand = None
    for b in sorted({brand_of(n) for n in phones['prod_name'].dropna()} - {None}, key=lambda s: -len(s)):
        if re.search(r'\b' + re.escape(norm(b)) + r'\b', q.n0):
            brand = b
            break
    line_hit = None
    for ml in sorted({model_line(n) for n in phones['prod_name'].dropna()}, key=lambda s: -len(s)):
        if re.search(r'\b' + re.escape(norm(ml)) + r'(?![a-z0-9])', q.n0):
            line_hit = ml
            break
    if q.has(r'pct|chiem|ti le') and (brand or q.has(r'may moi|may cu')):
        part = sub
        if brand:
            part = phones[phones['prod_name'].map(brand_of) == brand]
        if q.has(r'tien|doanh thu'):
            return A('percent', share(part['amount'].sum(), phones['amount'].sum()))
        return A('percent', share(part['qty'].sum(), phones['qty'].sum()))
    if line_hit and not q.has(r'tien|doanh thu'):
        x = phones[phones['prod_name'].map(model_line) == line_hit]
        return A('number', num(x['qty'].sum()))
    if q.has(r'trung binh|binh quan') and q.has(r'\bmay\b|chiec|dien thoai') and not q.has(r'\bdon\b'):
        if sub['qty'].sum() == 0:
            raise Abstain('no phones')
        return A('number', rnd(sub['amount'].sum() / sub['qty'].sum()))
    if q.has(r'(?:bao nhieu|may|so) (?:may|chiec|cai)\b|khong tinh phu kien|bo phu kien') and not q.has(r'tien|doanh thu'):
        return A('number', num(sub['qty'].sum()))
    return None


# ----------------------------------------------------------------------------
# mother & baby

def h_mombaby(q):
    biz = q.biz
    ng = biz.extra.get('nongoods')
    o0 = biz.orders
    if q.has(r'dang ky|thanh vien moi|khach moi'):
        c = biz.customers
        if c is None or 'created' not in c or c['created'].isna().all():
            return None
        p = need_period(q)
        cr = pd.to_datetime(c['created'])
        a, b = p.start, min(p.end, q.today)
        return A('number', int(((cr >= TS(a)) & (cr <= TS(b))).sum()))
    if ng is not None and q.has(r'\bdiem\b|vouche|phi ship|phi giao|tien ship'):
        p = need_period(q)
        o = orders_in(q, p)
        col = 'points_disc' if q.has(r'\bdiem\b') else ('voucher_disc' if q.has(r'vouche') else 'ship_fee')
        if col not in o:
            raise Abstain('no column')
        return A('number', rnd(o[col].fillna(0).abs().sum()))
    p_ = biz.products
    if p_ is None or 'size' not in p_ or p_['size'].isna().all():
        return None
    size_m = re.search(r'\b(?:size|so) (nb|xxl|xl|l|m|s|\d)\b', q.n0)
    brand = None
    for b in sorted({x for x in p_['brand'].dropna()}, key=lambda s: -len(s)):
        if re.search(r'\b' + re.escape(norm(b)) + r'\b', q.n0):
            brand = b
            break
    if not (size_m or brand) and not (q.has(r'\bhang nao\b') and superlative(q)):
        return None
    if not q.has(r'bich|goi|hop|lon|bao nhieu|pct|chiem'):
        return None
    p = need_period(q)
    o = orders_in(q, p)
    l = goods_lines(lines_of(q, o))
    info = p_.set_index('prod_key')
    diaper = q.has(r'\bbim\b|\bta\b')
    milk = q.has(r'\bsua\b')

    def in_family(k):
        if k not in info.index:
            return False
        c = norm(info.at[k, 'category'] or '')
        if diaper:
            return 'bim' in c or c.startswith('ta')
        if milk:
            return 'sua' in c
        return True
    fam = l[l['prod_key'].map(in_family)]
    sel = fam
    if size_m:
        want = size_m.group(1)
        sel = sel[sel['prod_key'].map(lambda k: norm(info.at[k, 'size'] or '').split()[-1:] == [want] if k in info.index else False)]
    if brand:
        sel = sel[sel['prod_key'].map(lambda k: info.at[k, 'brand'] == brand if k in info.index else False)]
    if q.has(r'\bhang nao\b') and superlative(q):
        b = sel.groupby(sel['prod_key'].map(lambda k: info.at[k, 'brand'] if k in info.index else None))
        s = b['amount'].sum() if q.has(r'tien|doanh thu') else b['qty_base'].sum()
        s = s[[k is not None for k in s.index]]
        if len(s) == 0:
            raise Abstain('no brand')
        keys, _ = rank_order(s, lambda k: str(k))
        return A('string', keys[0])
    if q.has(r'pct|chiem|ti le'):
        return A('percent', share(sel['qty_base'].sum(), fam['qty_base'].sum()))
    if q.has(r'doanh thu|tien'):
        return A('number', rnd(sel['amount'].sum()))
    return A('number', num(sel['qty_base'].sum()))


HANDLERS_SPECIAL = [h_repair, h_phone, h_mombaby]
