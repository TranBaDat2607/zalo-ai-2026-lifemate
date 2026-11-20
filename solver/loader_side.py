"""Side tables: payments, purchases, stock, prices, deliveries, lots, customers..."""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from loader import _num, cust_key, status_of, unit_key
from textnorm import norm
from values import clean_str, parse_bool, parse_datetime, parse_number, parse_qty


def _dt(v, t):
    return parse_datetime(v, t.year)


def load_side(biz, kinds, pidx):
    # ---- payments -------------------------------------------------------------------
    rows = []
    for t in kinds['payments']:
        for r in t.data:
            oid = clean_str(t.col('order_id', r))
            if not oid:
                continue
            rows.append(dict(order_id=oid, dt=_dt(t.col('paid_at', r) or t.col('date', r) or t.col('datetime', r), t),
                             amount=_num(t.col('pay_amount', r) or t.col('amount', r)),
                             method=clean_str(t.col('pay_method', r)), pay_id=clean_str(t.col('pay_id', r))))
    if rows:
        P = pd.DataFrame(rows)
        P['dt'] = pd.to_datetime(P['dt'])
        biz.payments = P

    # ---- customers ------------------------------------------------------------------
    rows = []
    for t in kinds['customers']:
        name_f = 'cust_name' if 'cust_name' in t.fmap else 'name'
        for r in t.data:
            rows.append(dict(cust_id=clean_str(t.col('cust_id', r)), name=clean_str(t.col(name_f, r)),
                             phone=clean_str(t.col('phone', r)), group=clean_str(t.col('cust_group', r)),
                             address=clean_str(t.col('address', r)),
                             created=_dt(t.col('cust_created', r), t)))
    if rows:
        biz.customers = pd.DataFrame(rows)

    # ---- purchases ------------------------------------------------------------------
    # a lot list (lot, expiry, qty) next to a priced purchase book is not a second purchase record
    ptabs = kinds['purchases']
    priced = [t for t in ptabs if any(x in t.fmap for x in ('rc_price', 'price', 'rc_amount', 'amount', 'rc_total'))]
    lot_only = [t for t in ptabs if t not in priced and 'lot_id' in t.fmap and ('prod_id' in t.fmap or 'prod_name' in t.fmap)]
    if priced and lot_only and any('prod_id' in t.fmap or 'prod_name' in t.fmap for t in priced):
        rows = []
        for t in lot_only:
            for r in t.data:
                pid, name = clean_str(t.col('prod_id', r)), clean_str(t.col('prod_name', r))
                rows.append(dict(prod_key=pidx.key_of(pid, name), lot_id=clean_str(t.col('lot_id', r)),
                                 rc_id=clean_str(t.col('rc_id', r)), expiry=_dt(t.col('expiry', r), t),
                                 qty_in=_num(t.col('rc_qty_base', r) or t.col('rc_qty', r) or t.col('qty', r)),
                                 cost=np.nan))
        biz.lots = pd.DataFrame(rows)
        biz.lots['expiry'] = pd.to_datetime(biz.lots['expiry']).dt.normalize()
        ptabs = [t for t in ptabs if t not in lot_only]
    heads = {}
    lines = []
    for t in ptabs:
        f = t.fmap
        has_prod = 'prod_name' in f or 'prod_id' in f
        for r in t.data:
            rc = clean_str(t.col('rc_id', r))
            d = dict(rc_id=rc,
                     date=_dt(t.col('rc_date', r) or t.col('date', r), t),
                     supplier=clean_str(t.col('supplier', r)), supplier_id=clean_str(t.col('supplier_id', r)),
                     rc_total=_num(t.col('rc_total', r)),
                     status=clean_str(t.col('rc_status', r) or t.col('status', r)),
                     note=clean_str(t.col('note', r)))
            if has_prod:
                pid = clean_str(t.col('prod_id', r))
                name = clean_str(t.col('prod_name', r))
                qn, qu = parse_qty(t.col('rc_qty', r) if 'rc_qty' in f else t.col('qty', r))
                unit = clean_str(t.col('rc_unit', r) or t.col('unit', r)) or qu
                d.update(prod_key=pidx.key_of(pid, name) if (pid or name) else None, prod_name=name,
                         unit=unit_key(unit), qty=qn if qn is not None else np.nan,
                         qty_base_given=_num(t.col('rc_qty_base', r)),
                         price=_num(t.col('rc_price', r) if 'rc_price' in f else t.col('price', r)),
                         amount=_num(t.col('rc_amount', r) if 'rc_amount' in f else t.col('amount', r)),
                         lot_id=clean_str(t.col('lot_id', r)), expiry=_dt(t.col('expiry', r), t),
                         line_no=t.col('rc_line', r) or t.col('line_no', r))
                lines.append(d)
            elif rc:
                heads[rc] = d
    if lines:
        PL = pd.DataFrame(lines)
        if heads:
            H = pd.DataFrame.from_dict(heads, orient='index')
            for c in ('date', 'supplier', 'supplier_id', 'rc_total', 'status', 'note'):
                if c in H:
                    m = PL['rc_id'].map(H[c])
                    if c in PL:
                        PL[c] = PL[c].where(PL[c].notna(), m)
                    else:
                        PL[c] = m
        # forward fill receipt-level blanks within receipt
        PL['rc_id'] = PL['rc_id'].ffill()
        for c in ('date', 'supplier', 'supplier_id', 'status', 'note'):
            if c in PL:
                PL[c] = PL.groupby('rc_id')[c].transform(lambda s: s.ffill().bfill())
        fac = []
        for k, u, qb in zip(PL['prod_key'], PL['unit'], PL['qty_base_given']):
            fac.append(biz.units.get(k, {}).get(u, 1.0))
        PL['qty_base'] = PL['qty_base_given'].where(PL['qty_base_given'].notna(), PL['qty'] * np.array(fac))
        PL['date'] = pd.to_datetime(PL['date']).dt.normalize()
        PL['cancelled'] = PL['status'].map(lambda s: bool(s) and bool(re.search(r'huy|cancel', norm(s))))
        biz.purchases = PL
    elif heads:
        H = pd.DataFrame.from_dict(heads, orient='index')
        H['date'] = pd.to_datetime(H['date']).dt.normalize()
        H['cancelled'] = H['status'].map(lambda s: bool(s) and bool(re.search(r'huy|cancel', norm(s))))
        biz.purchases = H.reset_index(drop=True)
    if heads:
        H = pd.DataFrame.from_dict(heads, orient='index')
        H['date'] = pd.to_datetime(H['date']).dt.normalize()
        H['cancelled'] = H['status'].map(lambda s: bool(s) and bool(re.search(r'huy|cancel', norm(s))))
        biz.extra['rc_heads'] = H

    # ---- stock ----------------------------------------------------------------------
    st = {}
    for t in kinds['stock']:
        name_f = 'prod_name' if 'prod_name' in t.fmap else 'name'
        for r in t.data:
            pid = clean_str(t.col('prod_id', r))
            name = clean_str(t.col(name_f, r))
            if not pid and not name:
                continue
            v = parse_number(t.col('stock', r))
            st[pidx.key_of(pid, name)] = v
    if st:
        biz.stock = st

    # ---- price history -----------------------------------------------------------------
    rows = []
    for t in kinds['prices']:
        name_f = 'prod_name' if 'prod_name' in t.fmap else 'name'
        for r in t.data:
            pid = clean_str(t.col('prod_id', r))
            name = clean_str(t.col(name_f, r))
            rows.append(dict(prod_key=pidx.key_of(pid, name), date=_dt(t.col('price_date', r), t),
                             price=_num(t.col('new_price', r)), old=_num(t.col('old_price', r))))
    if rows:
        PR = pd.DataFrame(rows)
        PR['date'] = pd.to_datetime(PR['date']).dt.normalize()
        biz.prices = PR.sort_values(['prod_key', 'date']).reset_index(drop=True)

    # ---- deliveries --------------------------------------------------------------------
    rows = []
    for t in kinds['deliveries']:
        for r in t.data:
            rows.append(dict(order_id=clean_str(t.col('order_id', r)), date=_dt(t.col('deliv_date', r), t),
                             batch=_num(t.col('deliv_batch', r)),
                             prod_key=pidx.key_of(clean_str(t.col('prod_id', r)), clean_str(t.col('prod_name', r))),
                             qty_base=_num(t.col('qty_base', r) or t.col('qty', r)),
                             plate=clean_str(t.col('plate', r)), driver=clean_str(t.col('driver', r))))
    if rows:
        D = pd.DataFrame(rows)
        D['order_id'] = D['order_id'].ffill()
        D['date'] = pd.to_datetime(D['date']).dt.normalize()
        biz.deliveries = D

    # ---- due dates -----------------------------------------------------------------------
    rows = []
    for t in kinds['due']:
        for r in t.data:
            rows.append(dict(order_id=clean_str(t.col('order_id', r)), days=_num(t.col('credit_days', r)),
                             due=_dt(t.col('due_date', r), t)))
    if rows:
        DU = pd.DataFrame(rows)
        DU['due'] = pd.to_datetime(DU['due']).dt.normalize()
        biz.due = DU

    # ---- lots of sold lines ------------------------------------------------------------------
    rows = []
    for t in kinds['line_lots']:
        for r in t.data:
            rows.append(dict(order_id=clean_str(t.col('order_id', r)), lot_id=clean_str(t.col('lot_id', r)),
                             line_no=t.col('line_no', r),
                             qty=_num(t.col('lot_qty', r) or t.col('qty', r)),
                             ret=_num(t.col('lot_ret_qty', r))))
    if rows:
        LL = pd.DataFrame(rows)
        LL['ret'] = LL['ret'].fillna(0)
        biz.line_lots = LL
        ret_orders = set(LL.loc[LL['ret'] > 0, 'order_id'])
        if ret_orders:
            o = biz.orders
            m = o['order_id'].isin(ret_orders) & (o['status'] != 'cancelled')
            o.loc[m, 'status'] = 'returned'

    # ---- expenses -----------------------------------------------------------------------------
    rows = []
    for t in kinds['expenses']:
        h = [norm(c) for c in t.header]
        jd = next((j for j, c in enumerate(h) if c.startswith('ngay')), None)
        ja = next((j for j, c in enumerate(h) if re.search(r'tien', c)), None)
        if ja is None:
            ja = next((j for j, c in enumerate(h) if re.search(r'\bchi\b', c) and not c.startswith('ngay')), None)
        jc = next((j for j, c in enumerate(h) if re.search(r'nhom|loai|khoan|muc', c)), None)
        jn = next((j for j, c in enumerate(h) if re.search(r'luu y|ghi chu|noi dung|dien giai', c)), None)
        for r in t.data:
            rows.append(dict(date=parse_datetime(r[jd], t.year) if jd is not None else None,
                             amount=_num(r[ja]) if ja is not None else np.nan,
                             category=clean_str(r[jc]) if jc is not None else None,
                             note=clean_str(r[jn]) if jn is not None else None))
    if rows:
        E = pd.DataFrame(rows)
        E['date'] = pd.to_datetime(E['date']).dt.normalize()
        biz.expenses = E

    # ---- cash per shift ------------------------------------------------------------------------
    rows = []
    for t in kinds['shifts']:
        h = [norm(c) for c in t.header]
        for r in t.data:
            d = dict(zip(h, r))
            rows.append(dict(date=parse_datetime(d.get('ngay'), t.year), shift=norm(d.get('ca')),
                             staff=clean_str(d.get('ma nv')), cash_in=_num(d.get('tien mat thu')),
                             refund=_num(d.get('tien mat hoan khach')), net=_num(d.get('tien mat rong cua ca'))))
    if rows:
        S = pd.DataFrame(rows)
        S['date'] = pd.to_datetime(S['date']).dt.normalize()
        biz.shifts = S
