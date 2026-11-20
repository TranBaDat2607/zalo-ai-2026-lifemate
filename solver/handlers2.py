"""Domain handlers: stock, purchases, debt, profit, lots, prices, deliveries."""
import datetime as dt
import re

import numpy as np
import pandas as pd

import periods as P
from handlers import (Flt, apply_line_filters, apply_order_filters, check_caps, cust_name_of, customer_mentioned,
                      money_values, superlative,
                      need_period, parse_filters, prep_orders, prod_code, rank_order, topk, unit_asked)
from qa import (A, Abstain, DAY, TS, Unanswerable, cap_customers, data_range, goods_lines, lines_of, need_lines,
                num, orders_in, pct, product_matches, rnd, share, window)
from textnorm import norm


def _day_end(d):
    return TS(d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def asof_date(q):
    """Date of 'den het ngay X' / 'cuoi ngay hom nay' (default today)."""
    for p in q.periods:
        if p.kind == 'day':
            return p.start
    return q.today


def one_product(q):
    pm = product_matches(q)
    if not pm:
        return None
    return pm[0][0]


def factor(biz, k, unit):
    if not unit:
        return 1.0
    return biz.units.get(k, {}).get(unit, 1.0)


def unit_named(q, k, triggers=r'mot|1|moi|bao nhieu|may|so|theo|tinh theo|bang'):
    """A unit of product k written after one of the trigger words."""
    biz = q.biz
    units = set(biz.units.get(k, {}).keys())
    bu = biz.extra.get('base_unit', {}).get(k)
    if bu:
        units.add(bu)
    units = {u for u in units if u}
    best = None
    for m in re.finditer(r'\b(?:' + triggers + r')\s+([a-z0-9]+(?: [a-z0-9]+){0,2})', q.n):
        w = m.group(1).split()
        for n in range(len(w), 0, -1):
            u = ' '.join(w[:n])
            if u in units:
                if best is None:
                    best = u
                break
    return best


def big_unit_of(biz, k):
    fu = {u: f for u, f in biz.units.get(k, {}).items() if f and f > 1}
    if not fu:
        return None, 1.0
    u = max(fu, key=fu.get)
    return u, fu[u]


# ----------------------------------------------------------------------------
# stock

def sold_qty_by_product(q, a, b, include_returned=True):
    """Base-unit quantity leaving stock for orders dated in [a, b] (not cancelled)."""
    biz = q.biz
    o = biz.orders
    m = (o['date'] >= TS(a)) & (o['date'] <= TS(b)) & (o['status'] != 'cancelled')
    if not include_returned:
        m &= o['status'] != 'returned'
    l = biz.lines
    l = l[l['order_id'].isin(set(o.loc[m, 'order_id'])) & ~l['is_fee']]
    l = l[l['qty'] > 0] if 'qty' in l else l
    return l.groupby('prod_key')['qty_base'].sum()


def purchased_qty_by_product(q, a, b):
    pu = q.biz.purchases
    if pu is None or 'prod_key' not in pu:
        return pd.Series(dtype=float)
    m = (pu['date'] >= TS(a)) & (pu['date'] <= TS(b)) & ~pu['cancelled'].fillna(False)
    return pu[m].groupby('prod_key')['qty_base'].sum()


def stock_at(q, d):
    """Series prod_key -> on-hand base qty at end of day d."""
    biz = q.biz
    sm = biz.stock_moves
    if sm is not None and len(sm):
        s = sm[sm['dt'] <= _day_end(d)].sort_values(['dt', 'seq'])
        last = s.groupby('prod_key')['after'].last()
        allp = pd.Series(0.0, index=sorted(set(sm['prod_key'])))
        allp.update(last)
        return allp
    if not biz.stock:
        if biz.line_lots is not None and (biz.lots is not None or (biz.purchases is not None and 'lot_id' in biz.purchases)):
            lots = lot_remaining(q, d)
            return lots.groupby('prod_key')['remaining'].sum()
        raise Unanswerable('no stock data')
    end = biz.data_end or biz.orders['date'].max().date()
    base = pd.Series(biz.stock, dtype=float).fillna(0.0)
    if d >= end:
        return base
    sold = sold_qty_by_product(q, d + DAY, end)
    bought = purchased_qty_by_product(q, d + DAY, end)
    # returns written as negative lines bring goods back
    st = base.add(sold, fill_value=0).sub(bought, fill_value=0)
    if biz.extra.get('neg_return_lines'):
        l = biz.lines
        o = biz.orders
        neg = l[(l['qty'] < 0)]
        if len(neg):
            od = o.set_index('order_id')['date']
            dd = neg['order_id'].map(od)
            sel = neg[(dd > TS(d)) & (dd <= TS(end))]
            st = st.add(sel.groupby('prod_key')['qty_base'].sum(), fill_value=0)
    return st


def h_stock(q):
    if not q.has(r'ton kho|trong kho|con lai|con bao nhieu|con may|con ton|ton cuoi|hang ton|tien hang nam trong kho|gia tri hang ton|\bton\b(?= (?:bao nhieu|may|duoc|con))|dut hang|het hang|chay hang'):
        return None
    if q.has(r'\bno\b|han dung|\bhet han\b|lo .*gan het han|khach|doanh thu|ban ra|ban duoc|tien ban'):
        return None
    biz = q.biz
    d = asof_date(q)
    if q.has(r'gia von|gia tri|tien hang') and not one_product(q):
        st = stock_at(q, d)
        p = biz.products
        cost = dict(zip(p['prod_key'], p['cost']))
        if all(pd.isna(cost.get(k)) for k in st.index):
            raise Unanswerable('no cost')
        val = sum(v * (cost.get(k) or 0) for k, v in st.items() if v > 0 and not pd.isna(cost.get(k)))
        return A('number', rnd(val))
    if q.has(r'het hang|dut hang|chay hang|bang 0|ton bang khong'):
        st = stock_at(q, d)
        return A('number', int((st <= 0).sum()))
    k = one_product(q)
    if k is None:
        return None
    st = stock_at(q, d)
    v = float(st.get(k, 0.0))
    if q.has(r'chan|nguyen|\ble\b'):
        m = re.search(r'([a-z0-9]+(?: [a-z0-9]+){0,2}) (?:chan|nguyen)\b', q.n)
        u = None
        if m:
            w = m.group(1).split()
            for n in range(1, len(w) + 1):
                cand = ' '.join(w[-n:])
                if cand in biz.units.get(k, {}):
                    u = cand
        if u is None:
            u, f = big_unit_of(biz, k)
        else:
            f = biz.units[k][u]
        if u is None or not f:
            return A('list', [0, num(v)])
        return A('list', [int(v // f), num(v - (v // f) * f)])
    u = unit_named(q, k)
    f = factor(biz, k, u) if u else 1.0
    return A('number', num(v / f))


# ----------------------------------------------------------------------------
# purchases

def need_purchases(q):
    pu = q.biz.purchases
    if pu is None or len(pu) == 0:
        raise Unanswerable('no purchase data')
    return pu


def receipts_in(q, p):
    pu = need_purchases(q)
    a, b = p.start, min(p.end, q.today)
    m = (pu['date'] >= TS(a)) & (pu['date'] <= TS(b)) & ~pu['cancelled'].fillna(False).astype(bool)
    return pu[m]


def receipt_totals(r):
    if 'rc_total' in r and r['rc_total'].notna().any():
        t = r.groupby('rc_id')['rc_total'].first()
        if 'amount' in r and r['amount'].notna().any():
            s = r.groupby('rc_id')['amount'].sum()
            t = t.where(t.notna(), s)
        return t
    if 'amount' in r and r['amount'].notna().any():
        return r.groupby('rc_id')['amount'].sum()
    raise Unanswerable('no purchase amounts')


def h_purchase(q):
    if not q.has(r'nhap|ncc|nha cung cap|danh hang|cat hang|lay hang|mua hang ve|dong hang'):
        return None
    if q.has(r'ton kho|trong kho') and not q.has(r'nhap hang'):
        return None
    biz = q.biz
    p = need_period(q)
    if q.has(r'\btru\b|\btru di\b|chenh lech|\bchenh\b') and q.has(r'\bban\b'):
        k = one_product(q)
        if k is None:
            return None
        r = receipts_in(q, p)
        qin = r.loc[r['prod_key'] == k, 'qty_base'].sum()
        o = orders_in(q, p)
        l = goods_lines(lines_of(q, o))
        qout = l.loc[l['prod_key'] == k, 'qty_base'].sum()
        u = unit_named(q, k)
        f = factor(biz, k, u) if u else 1.0
        return A('number', num((qin - qout) / f))
    r = receipts_in(q, p)
    k0 = one_product(q)
    if k0 is not None and not q.has(r'gia|tien|binh quan|trung binh') and unit_named(q, k0):
        u = unit_named(q, k0)
        rk = r[r['prod_key'] == k0]
        return A('number', num(rk['qty_base'].sum() / factor(biz, k0, u)))
    if q.has(r'gia nhap|nhap vao gia|binh quan|trung binh') and q.has(r'mot|1 |gia'):
        k = one_product(q)
        if k is None:
            return None
        if 'price' not in r or r['price'].isna().all():
            if 'amount' not in r or r['amount'].isna().all():
                raise Unanswerable('no purchase prices')
        rk = r[r['prod_key'] == k]
        if len(rk) == 0:
            raise Unanswerable('no purchases of item in period')
        amt = rk['amount'].where(rk['amount'].notna(), rk['qty'] * rk['price'])
        qb = rk['qty_base'].sum()
        u = unit_named(q, k, r'mot|1|moi|tren mot|tinh theo|theo')
        if u is None:
            units = rk['unit'].dropna().unique()
            u = units[0] if len(units) == 1 else None
        f = factor(biz, k, u) if u else 1.0
        if qb == 0:
            raise Abstain('zero qty')
        return A('number', rnd(amt.sum() / qb * f))
    if q.has(r'nha cung cap|ncc'):
        if q.has(r'chinh|nhieu phieu|nhieu lan|thuong xuyen|so phieu'):
            s = r.groupby(_sup_col(r))['rc_id'].nunique()
        else:
            t = receipt_totals(r)
            sup = r.groupby('rc_id')[_sup_col(r)].first()
            s = t.groupby(sup).sum()
        if len(s) == 0:
            raise Abstain('no receipts')
        keys, vals = rank_order(s, lambda k: str(k))
        name = keys[0]
        sup_names = biz.extra.get('suppliers', {})
        return A('string', sup_names.get(name, name))
    if q.has(r'bao nhieu tien|tong|gia tri|het bao nhieu|ton|bao nhieu'):
        return A('number', rnd(receipt_totals(r).sum()))
    return None


def _sup_col(r):
    if 'supplier' in r and r['supplier'].notna().any():
        return 'supplier'
    return 'supplier_id'


# ----------------------------------------------------------------------------
# profit

def cogs(l):
    if 'cost' not in l or l['cost'].isna().all():
        raise Unanswerable('no cost on sale lines')
    return (l['qty'] * l['cost'].fillna(0)).sum()


def h_profit(q):
    explicit = re.search(r'\b(?:lai|loi|laix) (?:gop|rong)\b|loi nhuan|\b(?:lai|loi) (?:nhieu |cao )?nhat\b', q.n)
    s = re.sub(r'con lai(?! (?:gop|rong))|cong lai|quay lai|mang lai|tro lai|ghe lai|lay lai|lai bao hanh|tinh lai|xem lai|kiem tra lai|thu lai|doi lai|tra lai|ban lai|mua lai|lai (?:cao|thap|tang|giam|nhieu|it) hon|roi lai', ' ', q.n)
    if not explicit and not re.search(r'\b(?:lai|loi|laix)\b', s):
        return None
    if q.has(r'khach|\bno\b'):
        return None
    biz = q.biz
    p = need_period(q)
    o = orders_in(q, p)
    l = lines_of(q, o)
    if q.has(r'mat hang|san pham|\bmon\b|\bhang\b.*nhat|nao') and q.has(r'nhat'):
        g = goods_lines(l)
        if 'cost' not in g or g['cost'].isna().all():
            raise Unanswerable('no cost')
        s = (g['amount'] - g['qty'] * g['cost'].fillna(0)).groupby(g['prod_key']).sum()
        keys, vals = rank_order(s, prod_code(biz))
        k = topk(q)
        names = [biz.prod_names.get(x, x) for x in keys]
        return A('list', names[:k]) if k else A('string', names[0])
    gp = float(l['amount'].sum()) - cogs(l)
    if q.has(r'rong|tru het chi phi|tru chi phi|sau chi phi'):
        e = biz.expenses
        if e is None:
            raise Unanswerable('no expenses')
        a, b = p.start, p.end
        ex = e[(e['date'] >= TS(a)) & (e['date'] <= TS(b))]['amount'].sum()
        return A('number', rnd(gp - ex))
    return A('number', rnd(gp))


# ----------------------------------------------------------------------------
# debt

def debt_orders(q):
    """Orders that can carry customer debt (not cancelled, not COD / marketplace)."""
    biz = q.biz
    prep_orders(biz)
    o = biz.orders
    o = o[o['status'] != 'cancelled']
    if 'pay_k' in o:
        o = o[o['pay_k'] != 'cod']
    if 'channel_k' in o:
        o = o[~o['channel_k'].isin(['marketplace', 'shopee', 'lazada', 'tiktok'])]
    return o


def debt_at(q, d):
    """Series order_id -> outstanding amount at end of day d."""
    biz = q.biz
    o = debt_orders(q)
    o = o[o['date'] <= TS(d)]
    pay = biz.payments
    if pay is not None and len(pay) and o['payable'].notna().any():
        paid = pay[pay['dt'] <= _day_end(d)].groupby('order_id')['amount'].sum()
        owed = o.set_index('order_id')['payable'] - paid.reindex(o['order_id']).fillna(0).values
        return owed[owed > 0.5]
    if 'debt' in o and o['debt'].notna().any():
        end = biz.data_end or biz.orders['date'].max().date()
        if d < end:
            raise Abstain('debt only known at end of data')
        owed = o.set_index('order_id')['debt']
        return owed[owed > 0.5]
    raise Unanswerable('no debt data')


def h_debt(q):
    if not q.has(r'\bno\b|cong no|mua chiu|ban chiu|ghi no|con thieu|chua tra|tra gop|thieu tien|khach thieu'):
        return None
    if q.has(r'ngay phai tra|han tra|han thanh toan|ngay den han|phai tra (?:truoc|trong|vao)'):
        return h_due(q)
    if q.has(r'tra gop|gop'):
        return None
    biz = q.biz
    if q.has(r'thu duoc|thu hoi|tra no|tra bot|thu no|tra duoc') and not q.has(r'con no|dang no'):
        return _debt_flow(q)
    if q.has(r'no them|ghi no them|mua chiu them'):
        return _debt_flow(q)
    d = asof_date(q)
    owed = debt_at(q, d)
    o = biz.orders.set_index('order_id')
    if q.has(r'tuoi|qua \d+ ngay|tren \d+ ngay|hon \d+ ngay|nhom|da dat'):
        age = (TS(d) - o.loc[owed.index, 'date']).dt.days
        rng = re.findall(r'(\d+)\s*[–\-]\s*(\d+)', q.raw.lower())
        if rng:
            out = []
            for a, b in rng:
                out.append(rnd(owed[(age >= int(a)) & (age <= int(b))].sum()))
            mm = re.search(r'(?:tren|qua|hon) (\d+)(?: ngay)?', q.n)
            if mm:
                out.append(rnd(owed[age > int(mm.group(1))].sum()))
            return A('list', out)
        mm = re.search(r'(?:qua|tren|hon) (\d+) ngay', q.n)
        if mm:
            return A('number', rnd(owed[age > int(mm.group(1))].sum()))
    ck = o.loc[owed.index, 'cust_key']
    cm = customer_mentioned(q)
    if cm is not None and not q.has(r'nhat|top'):
        return A('number', rnd(owed[ck == cm].sum()))
    if q.has(r'bao nhieu khach|may khach|so khach|bao nhieu nguoi|may nguoi|nhieu nguoi|dem khach|nhieu khach'):
        return A('number', int(ck.dropna().nunique()))
    if superlative(q) or topk(q):
        s = owed.groupby(ck).sum()
        s = s[s.index.notna()]
        keys, vals = rank_order(s, lambda k: str(k))
        k = topk(q) or 1
        names = cust_name_of(biz)
        rows = [[names.get(x, x), rnd(v)] for x, v in zip(keys[:k], vals[:k])]
        if q.has(r'moi nguoi|so no|kem|tung nguoi|bao nhieu'):
            return A('table', {'columns': ['khach_hang', 'so_tien_no'], 'rows': rows})
        if k > 1:
            return A('list', [r[0] for r in rows])
        return A('string', rows[0][0])
    return A('number', rnd(owed.sum()))


def _debt_flow(q):
    biz = q.biz
    p = need_period(q)
    pay = biz.payments
    if pay is None or len(pay) == 0:
        raise Unanswerable('no payment history')
    o = debt_orders(q)
    if 'pay_k' in o:
        deb = o[o['pay_k'] == 'debt']
    else:
        deb = o
    a, b = window(q, p)
    od = deb.set_index('order_id')['date']
    pp = pay[pay['order_id'].isin(od.index)].copy()
    pp['odate'] = pp['order_id'].map(od)
    pp['pdate'] = pp['dt'].dt.normalize()
    cust = customer_mentioned(q)
    if cust is not None:
        ids = set(deb.loc[deb['cust_key'] == cust, 'order_id'])
        pp = pp[pp['order_id'].isin(ids)]
        deb = deb[deb['cust_key'] == cust]
    later = pp[(pp['pdate'] > pp['odate']) & (pp['pdate'] >= TS(a)) & (pp['pdate'] <= TS(b))]
    collected = float(later['amount'].sum())
    inper = deb[(deb['date'] >= TS(a)) & (deb['date'] <= TS(b))]
    same = pp[pp['pdate'] == pp['odate']].groupby('order_id')['amount'].sum()
    new_debt = float((inper.set_index('order_id')['payable'] - same.reindex(inper['order_id']).fillna(0).values).clip(lower=0).sum())
    if q.has(r'no them') and q.has(r'tra'):
        return A('list', [rnd(new_debt), rnd(collected)])
    if q.has(r'no them|ghi no them|mua chiu them'):
        return A('number', rnd(new_debt))
    return A('number', rnd(collected))


# ----------------------------------------------------------------------------
# price on a date / price change effect

def h_price_on_date(q):
    if not q.has(r'gia ban|don gia|gia bao nhieu|ban .* gia|gia 1|gia mot'):
        return None
    if q.has(r'tang gia|len gia|doi gia|giam gia|so \d+ ngay|ngay ke tu'):
        return None
    k = one_product(q)
    if k is None:
        return None
    days = [p for p in q.periods if p.kind == 'day']
    if not days:
        return None
    d = days[0].start
    biz = q.biz
    l = need_lines(q)
    o = biz.orders
    ids = set(o.loc[o['date'] == TS(d), 'order_id'])
    ll = l[(l['order_id'].isin(ids)) & (l['prod_key'] == k)]
    if len(ll) == 0:
        if biz.prices is not None:
            pr = biz.prices[(biz.prices['prod_key'] == k) & (biz.prices['date'] <= TS(d))]
            if len(pr):
                return A('number', rnd(pr.iloc[-1]['price']))
        raise Abstain('no sale that day')
    per_base = ll['price'] / ll['qty_base'].where(ll['qty_base'] != 0) * ll['qty']
    per_base = per_base.dropna()
    if len(per_base) == 0:
        raise Abstain('no price')
    v = per_base.round(2).mode().iloc[0]
    return A('number', num(v))


def h_price_change(q):
    if not q.has(r'\btang gia\b|\blen gia\b|\bdoi gia\b|\bgia moi\b|\bgiam gia\b|\bha gia\b|\blen \d') or not q.has(r'ngay'):
        return None
    k = one_product(q)
    if k is None:
        return None
    m = re.search(r'(\d+) ngay', q.n)
    if not m:
        return None
    n = int(m.group(1))
    biz = q.biz
    days = [p for p in q.periods if p.kind == 'day']
    d = days[0].start if days else None
    if d is None:
        # find the change date from the price list: first time the price reached the given value
        mt = re.search(r'(?:lên|len|tăng lên|tang len)(?: giá| gia)?\s*(\d[\d.,]*)\s*(k|đ|d|đồng|dong|tr|nghìn|ngàn)?', q.raw.lower())
        target = []
        if mt:
            from values import parse_number
            unit = mt.group(2) or ''
            v = parse_number(mt.group(1) + ('k' if unit in ('k', 'nghìn', 'ngàn') else 'tr' if unit == 'tr' else ''))
            if v:
                target = [v]
        if not target:
            target = [v for _, v, u in money_values(q) if v >= 1000]
        if biz.prices is None or not target:
            raise Abstain('no change date')
        pr = biz.prices[biz.prices['prod_key'] == k].sort_values('date')
        prev = None
        for _, r in pr.iterrows():
            if prev is not None and r['price'] > prev and abs(r['price'] - target[0]) < 1:
                d = r['date'].date()
                break
            prev = r['price']
        if d is None:
            raise Abstain('change date not found')
    o = biz.orders
    l = need_lines(q)

    def qty(a, b):
        ids = set(o.loc[(o['date'] >= TS(a)) & (o['date'] <= TS(b)) & o['valid'], 'order_id'])
        return float(l.loc[l['order_id'].isin(ids) & (l['prod_key'] == k), 'qty_base'].sum())
    after = qty(d, d + dt.timedelta(days=n - 1))
    before = qty(d - dt.timedelta(days=n), d - DAY)
    return A('percent', pct(after, before))


# ----------------------------------------------------------------------------
# deliveries

def h_delivery(q):
    if not q.has(r'\bdot\b|\bchuyen hang\b|\bluot giao\b|xe .*\bcho\b .*(?:dot|chuyen)|chua giao|da giao (?:xong|het)'
                 r'|(?:da )?(?:cho|cho di|giao) (?:duoc )?(?:bao nhieu|may)'):
        return None
    biz = q.biz
    D = biz.deliveries
    if D is None:
        if q.has(r'dot giao|dot hang'):
            raise Unanswerable('no delivery data')
        return None
    p = need_period(q)
    a, b = p.start, min(p.end, q.today)
    if q.has(r'(\d+) dot tro len|tu (\d+) dot|it nhat (\d+) dot|thanh (\d+) dot'):
        m = re.search(r'(?:tu |it nhat |thanh )?(\d+) dot', q.n)
        k = int(m.group(1))
        o = biz.orders
        ids = set(o.loc[(o['date'] >= TS(a)) & (o['date'] <= TS(b)), 'order_id'])
        dd = D[D['order_id'].isin(ids) & (D['date'] <= TS(q.today))]
        c = dd.groupby('order_id')['batch'].nunique()
        return A('number', int((c >= k).sum()))
    k = one_product(q)
    if k is not None and q.has(r'don dat|dat (?:trong )?thang|cho khach dat'):
        o = biz.orders
        ids = set(o.loc[(o['date'] >= TS(p.start)) & (o['date'] <= TS(min(p.end, q.today))) & (o['status'] != 'cancelled'), 'order_id'])
        dd = D[D['order_id'].isin(ids) & (D['prod_key'] == k) & (D['date'] <= TS(q.today))]
        u = unit_named(q, k)
        f = factor(biz, k, u) if u else 1.0
        return A('number', num(dd['qty_base'].sum() / f))
    dd = D[(D['date'] >= TS(a)) & (D['date'] <= TS(b))]
    return A('number', int(dd[['order_id', 'batch']].drop_duplicates().shape[0]))


# ----------------------------------------------------------------------------
# unsold items

def h_unsold(q):
    raw = q.raw.lower()
    e_word = re.search(r'(?<![\wà-ỹ])ế(?![\wà-ỹ])|\bees\b|\bee\b(?= h)', raw) is not None
    if not (e_word or q.has(r'khong ban (?:ra |duoc )*(?:duoc )?(?:cai|mon|lan|chiec|hop|goi|san pham|mat hang|mot)|khong ai mua'
                            r'|ban khong duoc (?:cai|mon)|chua ban duoc (?:cai|mon)|khong co don nao mua')):
        return None
    if not q.has(r'mat hang|san pham|\bmon\b|\bhang\b'):
        return None
    p = need_period(q)
    biz = q.biz
    o = biz.orders
    l = goods_lines(need_lines(q))
    a, b = window(q, p)
    before_ids = set(o.loc[(o['date'] < TS(a)) & o['valid'], 'order_id'])
    in_ids = set(o.loc[(o['date'] >= TS(a)) & (o['date'] <= TS(b)) & o['valid'], 'order_id'])
    sold_before = set(l.loc[l['order_id'].isin(before_ids), 'prod_key'])
    sold_in = set(l.loc[l['order_id'].isin(in_ids), 'prod_key'])
    unsold = sold_before - sold_in
    if q.has(r'liet ke|la nhung|nhung (?:mat hang|san pham|mon|hang) nao|ke ten|ten cac') and not q.has(r'bao nhieu|\bmay\b|so luong'):
        code = prod_code(biz)
        ks = sorted(unsold, key=code)
        return A('list', [biz.prod_names.get(k, k) for k in ks])
    return A('number', len(unsold))


# ----------------------------------------------------------------------------
# lots / expiry

def lot_remaining(q, d):
    biz = q.biz
    lots = biz.lots
    if lots is None:
        pu = biz.purchases
        if pu is None or 'expiry' not in pu or pu['expiry'].isna().all():
            raise Unanswerable('no lot / expiry data')
        lots = pu.rename(columns={'qty_base': 'qty_in'})
        lots = lots[~lots['cancelled'].fillna(False).astype(bool)]
        lots = lots[['prod_key', 'lot_id', 'expiry', 'qty_in', 'date'] + (['price'] if 'price' in lots else [])].copy()
        if 'price' in lots:
            f = [factor(biz, k, u) for k, u in zip(pu.loc[lots.index, 'prod_key'], pu.loc[lots.index, 'unit'])]
            lots['cost'] = lots['price'] / np.array(f)
        else:
            lots['cost'] = np.nan
    else:
        lots = lots.copy()
        pu = biz.purchases
        if pu is not None and 'rc_id' in lots:
            rd = pu.groupby('rc_id')['date'].first()
            lots['date'] = lots['rc_id'].map(rd)
        if lots['cost'].isna().all() and pu is not None and 'lot_id' in pu and 'price' in pu:
            pl = pu.dropna(subset=['lot_id'])
            f = np.array([factor(biz, k, u) for k, u in zip(pl['prod_key'], pl['unit'])])
            lots['cost'] = lots['lot_id'].map(dict(zip(pl['lot_id'], pl['price'] / f)))
    lots = lots[lots['date'].isna() | (lots['date'] <= TS(d))] if 'date' in lots else lots
    ll = biz.line_lots
    if ll is None:
        raise Unanswerable('no lot sales data')
    o = biz.orders
    od = o.set_index('order_id')
    ok = od.index[(od['date'] <= TS(d)) & (od['status'] != 'cancelled')]
    used = ll[ll['order_id'].isin(set(ok))]
    out_q = used.groupby('lot_id')['qty'].sum() - used.groupby('lot_id')['ret'].sum()
    lots['remaining'] = lots['qty_in'] - lots['lot_id'].map(out_q).fillna(0)
    lots['expiry'] = pd.to_datetime(lots['expiry']).dt.normalize()
    return lots


def h_expiry(q):
    import unicodedata
    raw = unicodedata.normalize('NFC', q.raw.lower())
    if not (re.search(r'hạn dùng|hết hạn|hạn sử dụng|\bhsd\b|\blô\b|date', raw) or
            (q.has(r'han dung|het han|han su dung|\bhsd\b|\blo\b') and not re.search(r'[à-ỹ]', raw))):
        return None
    if q.has(r'ngay phai tra|ban chiu|mua chiu|cong no|\bno\b'):
        return None
    biz = q.biz
    d = q.today
    lots = lot_remaining(q, d)
    live = lots[lots['remaining'] > 0]
    if q.has(r'gan het han nhat|som nhat|het han som|can date|can han|sap het han nhat|gan date'):
        k = one_product(q)
        if k is None:
            return None
        x = live[(live['prod_key'] == k) & (live['expiry'] >= TS(d))]
        if len(x) == 0:
            raise Abstain('no live lot')
        return A('date', x['expiry'].min().strftime('%Y-%m-%d'))
    m = re.search(r'(\d+) ngay toi', q.n)
    rng = [p for p in q.periods if p.kind in ('range', 'day')]
    days = [p for p in q.periods if p.kind == 'day']
    if rng and rng[0].kind == 'range':
        a, b = rng[0].start, rng[0].end
    elif m:
        a, b = d, d + dt.timedelta(days=int(m.group(1)))
    elif days and q.has(r'tu (?:nay|hom nay|bay gio|gio)\b'):
        a, b = d, max(p.start for p in days)
    else:
        return None
    x = live[(live['expiry'] >= TS(a)) & (live['expiry'] <= TS(b))]
    if q.has(r'tien von|gia tri|bao nhieu tien|\bvon\b|dong bao nhieu|dong tien'):
        if x['cost'].isna().all():
            raise Unanswerable('no lot cost')
        return A('number', rnd((x['remaining'] * x['cost']).sum()))
    if q.has(r'(?:bao nhieu|may|so) lo\b'):
        return A('number', int(x['lot_id'].nunique()))
    if q.has(r'tien von|gia tri|bao nhieu tien|\bvon\b'):
        if x['cost'].isna().all():
            raise Unanswerable('no lot cost')
        return A('number', rnd((x['remaining'] * x['cost']).sum()))
    if q.has(r'bao nhieu|may|so') and not q.has(r'nhung|nao|liet ke'):
        return A('number', int(x['prod_key'].nunique()))
    first = x.groupby('prod_key')['expiry'].min()
    code = prod_code(biz)
    ks = sorted(first.index, key=lambda k: (first[k], code(k)))
    return A('list', [biz.prod_names.get(k, k) for k in ks])


def h_due(q):
    """Credit orders by due date (sổ hạn nợ)."""
    biz = q.biz
    if biz.due is None:
        raise Unanswerable('no due dates')
    d = q.today
    o = biz.orders
    ok = o[(o['status'] != 'cancelled') & (o['date'] <= TS(d))]['order_id']
    du = biz.due[biz.due['order_id'].isin(set(ok))]
    months = [p for p in q.periods if p.kind in ('month', 'quarter', 'range', 'week')]
    if months:
        p = months[0]
        sel = du[(du['due'] >= TS(p.start)) & (du['due'] <= TS(p.end))]
    elif q.has(r'sau hom nay|muon hon hom nay|con sau|chua toi han|chua den han|sau ngay hom nay'):
        sel = du[du['due'] > TS(d)]
    elif q.has(r'qua han|tre han|truoc hom nay'):
        sel = du[du['due'] < TS(d)]
    elif q.has(r'dung hom nay|den han hom nay|han hom nay'):
        sel = du[du['due'] == TS(d)]
    else:
        return None
    return A('number', int(sel['order_id'].nunique()))


def h_peak_hour(q):
    if not q.has(r'gio cao diem|may gio|khung gio|gio nao|gio dong|dong khach nhat .*gio|gio .*dong khach'):
        return None
    biz = q.biz
    o = biz.orders
    if o['dt'].isna().mean() > 0.5:
        raise Unanswerable('no time of day')
    if q.periods:
        o = orders_in(q, q.periods[0])
    else:
        o = o[o['valid'] & (o['date'] <= TS(q.today))]
    h = o['dt'].dropna().dt.hour
    if len(h) == 0:
        raise Abstain('no orders')
    c = h.value_counts()
    best = c[c == c.max()].index.min()
    return A('number', int(best))


HANDLERS2 = [h_peak_hour, h_price_change, h_price_on_date, h_expiry, h_stock, h_debt, h_profit, h_purchase, h_delivery, h_unsold]
