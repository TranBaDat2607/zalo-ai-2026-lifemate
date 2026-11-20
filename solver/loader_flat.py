"""Assembly of canonical tables from csv / xlsx exports."""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from loader import (ORDER_FIELDS, _num, cust_key, is_retail, resolve_units, status_of, unit_key,
                    _parse_factor)
from textnorm import norm
from values import clean_str, parse_bool, parse_datetime, parse_number, parse_qty

PROD_FIELDS = ('prod_name', 'prod_id')


def classify(t):
    f = t.fmap
    fn = norm(t.fname + ' ' + (t.sheet or ''))
    kinds = []
    if 'deliv_date' in f or 'deliv_batch' in f:
        return ['deliveries']
    if 'due_date' in f or 'credit_days' in f:
        return ['due']
    if 'lot_id' in f and 'order_id' in f:
        return ['line_lots']
    if 'rc_id' in f or 'rc_date' in f:
        return ['purchases']
    if 'price_date' in f:
        return ['prices']
    if re.search(r'chi ?phi', fn):
        return ['expenses']
    if re.search(r'bao ?cao ?ca', fn):
        return ['shifts']
    if 'factor' in f or 'unit_rule' in f:
        kinds.append('units')
    if 'order_id' in f and ('pay_id' in f or 'paid_at' in f) and not any(p in f for p in PROD_FIELDS):
        return ['payments']
    has_prod = any(p in f for p in PROD_FIELDS)
    if has_prod and ('order_id' in f or 'date' in f) and any(x in f for x in ('qty', 'amount', 'qty_base')):
        return ['lines']
    if 'order_id' in f and not has_prod:
        return ['orders']
    if 'ingredient' in f or 'rx' in f:
        kinds.append('drugs')
    if 'stock' in f:
        kinds.append('stock')
    if re.search(r'khach ?hang', fn) or ('cust_id' in f and 'order_id' not in f):
        if not has_prod:
            kinds.append('customers')
    if (has_prod or 'name' in f) and any(x in f for x in ('category', 'cur_price', 'cost', 'base_unit', 'active')):
        if 'customers' not in kinds:
            kinds.append('products')
    return kinds or ['other']


# ----------------------------------------------------------------------------

def _rows_to_records(t, fields):
    """Raw values for the given fields, with forward fill of order-level blanks."""
    out = []
    for r in t.data:
        out.append({f: t.col(f, r) for f in fields if f in t.fmap})
    return out


def _parse_dt(v, year):
    d = parse_datetime(v, year)
    return d


class ProductIndex:
    """Maps the many spellings of a product to one key."""

    def __init__(self):
        self.by_id = {}
        self.by_key = {}
        self.names = defaultdict(Counter)
        self.info = {}

    def key_of(self, pid=None, name=None):
        if pid is not None:
            pid = str(pid).strip()
            if pid in self.by_id:
                return self.by_id[pid]
        if name is not None:
            k = norm(name)
            if k in self.by_key:
                return self.by_key[k]
            if pid:
                self.by_id[pid] = k
            return k
        return pid

    def add(self, pid, name, **info):
        k = norm(name) if name else (str(pid).strip() if pid else None)
        if k is None:
            return None
        if pid:
            pid = str(pid).strip()
            if pid in self.by_id:
                k = self.by_id[pid]
            self.by_id[pid] = k
        if name:
            self.by_key.setdefault(norm(name), k)
            self.names[k][clean_str(name)] += 1
        d = self.info.setdefault(k, {})
        if pid:
            d.setdefault('prod_id', pid)
        for a, b in info.items():
            if b is not None and (a not in d or d[a] is None):
                d[a] = b
        return k


def assemble(biz, tables):
    rd = biz.readme
    kinds = defaultdict(list)
    for t in tables:
        t.kinds = classify(t)
        for k in t.kinds:
            kinds[k].append(t)
    biz.extra['tables'] = [(t.fname, t.sheet, t.kinds, sorted(t.fmap)) for t in tables]
    for t in tables:
        for f in t.fmap:
            if f.startswith('EMPTY_'):
                biz.empty_cols.add(f[6:])

    pidx = ProductIndex()
    # ---- catalogue -----------------------------------------------------------------
    for t in kinds['products']:
        name_f = 'prod_name' if 'prod_name' in t.fmap else 'name'
        for r in t.data:
            name = clean_str(t.col(name_f, r))
            pid = clean_str(t.col('prod_id', r))
            if not name and not pid:
                continue
            pidx.add(pid, name, category=clean_str(t.col('category', r)),
                     base_unit=clean_str(t.col('base_unit', r) or t.col('unit', r)),
                     cur_price=parse_number(t.col('cur_price', r)), cost=parse_number(t.col('cost', r)),
                     active=parse_bool(t.col('active', r)), created=parse_datetime(t.col('prod_created', r)),
                     brand=clean_str(t.col('brand', r)), size=clean_str(t.col('size', r)),
                     aliases=clean_str(t.col('alias_names', r)), unit_rule=clean_str(t.col('unit_rule', r)))
    for t in kinds['drugs']:
        for r in t.data:
            pid = clean_str(t.col('prod_id', r))
            name = clean_str(t.col('prod_name', r) or t.col('name', r))
            k = pidx.add(pid, name)
            if k is None:
                continue
            d = pidx.info[k]
            d['ingredient'] = clean_str(t.col('ingredient', r))
            d['rx'] = parse_bool(t.col('rx', r))

    # ---- lines ---------------------------------------------------------------------
    line_tables = kinds['lines']
    recs = []
    seq = 0
    topping_mode = rd.has('topping') and rd.has('ghi chung ô')
    for t in line_tables:
        f = t.fmap
        prev_oid = None
        prev_vals = {}
        seen_rows = set()
        synth = 'order_id' not in f
        for r in t.data:
            raw_oid = clean_str(t.col('order_id', r))
            if synth:
                seq += 1
                oid = f'R{seq:07d}'
            elif raw_oid is None:
                oid = prev_oid
                if oid is None:
                    continue
            else:
                oid = raw_oid
            if oid != prev_oid:
                prev_vals = {}
            # duplicate export rows of the same order
            sig = (oid, tuple('' if c is None else str(c) for c in r))
            if not synth and sig in seen_rows:
                continue
            seen_rows.add(sig)
            rec = {'order_id': oid, '_src': t.fname, '_row': seq}
            for fld in ORDER_FIELDS:
                if fld in f:
                    v = t.col(fld, r)
                    if v is None and not synth:
                        v = prev_vals.get(fld)
                    else:
                        prev_vals[fld] = v
                    rec[fld] = v
            for fld in ('prod_id', 'prod_name', 'unit', 'qty', 'qty_base', 'price', 'line_disc', 'amount',
                        'cost', 'line_no', 'category', 'imei', 'discount'):
                if fld in f:
                    rec[fld] = t.col(fld, r)
            rec['_year'] = t.year
            prev_oid = oid
            recs.append(rec)
    if not recs:
        if kinds['orders']:
            _orders_only(biz, kinds, pidx)
            return
        raise RuntimeError(f'{biz.bid}: no line tables found {[(t.fname, t.kinds) for t in tables]}')

    L = pd.DataFrame.from_records(recs)
    # parse values
    yr = L['_year']
    L['date_p'] = [parse_datetime(v, y) for v, y in zip(L.get('date', [None] * len(L)), yr)]
    if 'datetime' in L:
        L['dt_p'] = [parse_datetime(v, y) for v, y in zip(L['datetime'], yr)]
    elif 'time' in L:
        L['dt_p'] = [parse_datetime(v, y) for v, y in zip(L['time'], yr)]
    else:
        L['dt_p'] = None
    qn = [parse_qty(v) for v in L['qty']] if 'qty' in L else [(np.nan, None)] * len(L)
    L['qty_n'] = [q[0] if q[0] is not None else np.nan for q in qn]
    L['qty_unit'] = [q[1] for q in qn]
    for fld in ('qty_base', 'price', 'line_disc', 'amount', 'cost', 'goods_total', 'order_disc', 'ship_fee',
                'payable', 'paid', 'debt', 'tradein', 'cust_paid_net', 'discount', 'guests'):
        if fld in L:
            L[fld + '_n'] = L[fld].map(_num)
    # product identity
    keys, names = [], []
    for pid, name in zip(L.get('prod_id', [None] * len(L)), L.get('prod_name', [None] * len(L))):
        pid = clean_str(pid)
        name = clean_str(name)
        keys.append(pidx.key_of(pid, name) if (pid or name) else None)
        names.append(name)
    L['prod_key'] = keys
    for k, n in zip(keys, names):
        if k and n:
            pidx.names[k][n] += 1

    # ---- units ---------------------------------------------------------------------
    raw_units = defaultdict(list)
    for prod, big, fac, base in rd.units:
        raw_units[pidx.key_of(None, prod)].append((unit_key(big), fac, None))
    for t in kinds['units']:
        for r in t.data:
            name = clean_str(t.col('prod_name', r) or t.col('name', r))
            pid = clean_str(t.col('prod_id', r))
            if not name and not pid:
                continue
            k = pidx.key_of(pid, name)
            if 'unit_rule' in t.fmap:
                rule = clean_str(t.col('unit_rule', r))
                if rule:
                    for m in re.finditer(r'1\s+(.+?)\s*=\s*([\d.,]+)\s*([^\d,;]+)', rule):
                        raw_units[k].append((unit_key(m.group(1)), parse_number(m.group(2)), unit_key(m.group(3))))
                continue
            big = clean_str(t.col('big_unit', r) or t.col('unit', r))
            fac, inu = _parse_factor(t.col('factor', r), None)
            base = clean_str(t.col('base_unit', r))
            if base:
                pidx.info.setdefault(k, {}).setdefault('base_unit', base)
            if big and fac:
                raw_units[k].append((unit_key(big), fac, inu))
    for k, d in pidx.info.items():
        rule = d.get('unit_rule')
        if rule:
            for m in re.finditer(r'1\s+(.+?)\s*=\s*([\d.,]+)\s*([^\d,;]+)', rule):
                raw_units[k].append((unit_key(m.group(1)), parse_number(m.group(2)), unit_key(m.group(3))))
    biz.units = resolve_units(raw_units)
    # base units: catalogue, else smallest unit seen with qty_base == qty
    for k, d in pidx.info.items():
        if d.get('base_unit'):
            biz.units.setdefault(k, {})[unit_key(d['base_unit'])] = 1.0
    # unit of the line: explicit unit column, else unit text in the qty cell
    L['unit_k'] = [unit_key(u) if u not in (None, '') else unit_key(qu) if qu else ''
                   for u, qu in zip(L.get('unit', [None] * len(L)), L['qty_unit'])]

    # qty in base units
    if 'qty_base_n' not in L or L['qty_base_n'].isna().all():
        fac = []
        for k, u in zip(L['prod_key'], L['unit_k']):
            fu = biz.units.get(k, {})
            fac.append(fu.get(u, 1.0))
        L['qty_base_n'] = L['qty_n'] * np.array(fac)
    else:
        miss = L['qty_base_n'].isna()
        if miss.any():
            fac = [biz.units.get(k, {}).get(u, 1.0) for k, u in zip(L.loc[miss, 'prod_key'], L.loc[miss, 'unit_k'])]
            L.loc[miss, 'qty_base_n'] = L.loc[miss, 'qty_n'] * np.array(fac)
    # base unit per product: catalogue, else unit of lines where qty_base == qty
    base_unit = {k: d.get('base_unit') for k, d in pidx.info.items() if d.get('base_unit')}
    same = L[(L['qty_n'] == L['qty_base_n']) & L['prod_key'].notna()]
    for k, u in same.groupby('prod_key')['unit_k'].agg(lambda s: s.value_counts().index[0] if len(s) else '').items():
        if k not in base_unit and u:
            base_unit[k] = u
    for k, fu in biz.units.items():
        if k not in base_unit:
            ones = [u for u, f in fu.items() if f == 1.0]
            if ones:
                base_unit[k] = ones[0]
    biz.extra['base_unit'] = {k: unit_key(v) for k, v in base_unit.items()}

    if 'amount_n' not in L:
        L['amount_n'] = L['qty_n'] * L.get('price_n', np.nan)
    L['amount_n'] = L['amount_n'].fillna(L['qty_n'] * L.get('price_n', np.nan))

    # ---- toppings written in the item cell (coffee) -----------------------------------
    if topping_mode:
        L = _split_toppings(L, pidx)

    from loader_special import nongoods_kind
    ng = L['prod_name'].map(lambda x: nongoods_kind(x) if isinstance(x, str) else None) if 'prod_name' in L else None
    if ng is not None and ng.notna().any():
        NG = L[ng.notna()].assign(kind=ng[ng.notna()])
        agg = NG.groupby(['order_id', 'kind'])['amount_n'].sum().unstack(fill_value=0.0)
        biz.extra['nongoods'] = agg
        L = L[ng.isna()].copy()
    biz.lines_raw_count = len(L)
    lines = pd.DataFrame({
        'order_id': L['order_id'],
        'prod_key': L['prod_key'],
        'prod_name': L.get('prod_name'),
        'unit': L['unit_k'],
        'qty': L['qty_n'],
        'qty_base': L['qty_base_n'],
        'price': L.get('price_n', np.nan),
        'line_disc': L.get('line_disc_n', np.nan),
        'amount': L['amount_n'],
        'cost': L.get('cost_n', np.nan),
        'line_no': L.get('line_no'),
        'category_line': L.get('category'),
        'imei': L.get('imei'),
    })
    lines['date'] = [d.date() if d else None for d in L['date_p']]
    lines['is_fee'] = lines['prod_key'].isin(['phi phuc vu'])

    # ---- orders from lines -----------------------------------------------------------
    og = {}
    order_cols = [c for c in ORDER_FIELDS if c in L]
    first = L.groupby('order_id', sort=False).first()
    O = pd.DataFrame(index=first.index)
    O['order_id'] = O.index
    O['date'] = [d.date() if isinstance(d, (pd.Timestamp,)) or hasattr(d, 'date') else None
                 for d in first['date_p']] if 'date_p' in first else None
    # orders whose first line has no date: use any line date
    O['dt'] = first['dt_p'] if 'dt_p' in first else None
    if O['date'].isna().any():
        dd = L.dropna(subset=['date_p']).groupby('order_id')['date_p'].first()
        O['date'] = O['date'].fillna(dd.map(lambda d: d.date()))
    if O['date'].isna().any() and O['dt'].notna().any():
        O.loc[O['date'].isna(), 'date'] = O.loc[O['date'].isna(), 'dt'].map(lambda d: d.date() if d else None)
    # if datetime present but date missing in some rows, use dt
    for c in ('status', 'cust_id', 'cust_name', 'cust_group', 'staff_id', 'channel', 'pay_method', 'note',
              'table', 'area', 'phone', 'address'):
        if c in first:
            O[c] = first[c].map(clean_str)
    for c in ('goods_total', 'order_disc', 'ship_fee', 'payable', 'paid', 'debt', 'tradein', 'cust_paid_net', 'guests'):
        if c + '_n' in first:
            O[c] = first[c + '_n']
    O = O.rename(columns={'staff_id': 'staff'})
    # status
    if 'status' in O:
        O['status_raw'] = O['status']
        O['status'] = O['status'].map(status_of)
    elif 'note' in O:
        from loader_special import status_from_note
        O['status'] = O['note'].map(status_from_note)
    else:
        O['status'] = 'completed'
    # returns written as negative quantity lines
    neg = lines.loc[lines['qty'] < 0, 'order_id'].unique()
    if len(neg):
        O.loc[O.index.isin(neg) & (O['status'] != 'cancelled'), 'status'] = 'returned'
        biz.extra['neg_return_lines'] = True
    if 'nongoods' in biz.extra:
        agg = biz.extra['nongoods']
        for c, dst in (('voucher', 'voucher_disc'), ('points', 'points_disc'), ('ship', 'ship_fee')):
            if c in agg:
                O[dst] = agg[c].reindex(O.index).fillna(0.0).abs()
    rev = lines[~lines['qty'].lt(0)].groupby('order_id')['amount'].sum()
    O['rev'] = rev.reindex(O.index).fillna(0.0)

    # ---- separate order tables (invoices) ---------------------------------------------
    for t in kinds['orders']:
        _merge_order_table(O, t, biz)
    if 'note' in O and rd.has('ghi nợ thì ghi khách đã đưa'):
        from loader_special import debt_from_note
        owed = [debt_from_note(n, p) for n, p in zip(O['note'], O.get('payable', pd.Series(np.nan, index=O.index)))]
        O['debt'] = owed
        biz.extra['debt_from_notes'] = True
    biz.orders = O.reset_index(drop=True)
    biz.lines = lines

    # ---- products table --------------------------------------------------------------
    rows = []
    cat_line = lines.dropna(subset=['category_line']).groupby('prod_key')['category_line'].agg(
        lambda s: s.value_counts().index[0]) if lines['category_line'].notna().any() else {}
    for k in set(pidx.info) | set(lines['prod_key'].dropna().unique()):
        d = pidx.info.get(k, {})
        nm = pidx.names.get(k)
        name = nm.most_common(1)[0][0] if nm else d.get('prod_id', k)
        cat = d.get('category') or (cat_line.get(k) if hasattr(cat_line, 'get') else None)
        rows.append(dict(prod_key=k, prod_id=d.get('prod_id'), name=name, category=cat,
                         base_unit=biz.extra['base_unit'].get(k), cur_price=d.get('cur_price'), cost=d.get('cost'),
                         active=d.get('active'), rx=d.get('rx'), ingredient=d.get('ingredient'),
                         brand=d.get('brand'), size=d.get('size'), aliases=d.get('aliases'),
                         created=d.get('created')))
    biz.products = pd.DataFrame(rows)
    biz.prod_names = {r['prod_key']: r['name'] for r in rows}
    biz.prod_spellings = {k: list(v) for k, v in pidx.names.items()}
    biz.pidx = pidx

    _load_side_tables(biz, kinds, pidx)


def _merge_order_table(O, t, biz):
    f = t.fmap
    recs = {}
    for r in t.data:
        oid = clean_str(t.col('order_id', r))
        if not oid:
            continue
        d = {}
        for c in ('date', 'datetime', 'status', 'cust_id', 'cust_name', 'cust_group', 'staff_id', 'channel',
                  'pay_method', 'note', 'goods_total', 'discount', 'order_disc', 'ship_fee', 'payable', 'paid',
                  'debt', 'vat', 'table', 'guests'):
            if c in f:
                d[c] = t.col(c, r)
        recs[oid] = d
    if not recs:
        return
    T = pd.DataFrame.from_dict(recs, orient='index')
    idx = T.index.intersection(O.index)
    new = T.index.difference(O.index)
    if 'date' in T:
        dates = T['date'].map(lambda v: parse_datetime(v, t.year))
        T['date_p'] = dates
    if 'datetime' in T:
        T['dt_p'] = T['datetime'].map(lambda v: parse_datetime(v, t.year))
    for c, dst in (('goods_total', 'goods_total'), ('discount', 'order_disc'), ('order_disc', 'order_disc'),
                   ('ship_fee', 'ship_fee'), ('payable', 'payable'), ('paid', 'paid'), ('debt', 'debt'),
                   ('vat', 'vat'), ('guests', 'guests')):
        if c in T:
            O.loc[idx, dst] = T.loc[idx, c].map(_num).values
    for c, dst in (('cust_id', 'cust_id'), ('cust_name', 'cust_name'), ('cust_group', 'cust_group'),
                   ('staff_id', 'staff'), ('channel', 'channel'), ('pay_method', 'pay_method'), ('note', 'note'),
                   ('table', 'table')):
        if c in T:
            O.loc[idx, dst] = T.loc[idx, c].map(clean_str).values
    if 'status' in T:
        st = T.loc[idx, 'status'].map(status_of)
        keep_ret = O.loc[idx, 'status'].eq('returned')
        O.loc[idx, 'status'] = np.where(keep_ret & st.ne('cancelled').values, 'returned', st.values)
        O.loc[idx, 'status_raw'] = T.loc[idx, 'status'].values
    if 'date_p' in T:
        O.loc[idx, 'date'] = T.loc[idx, 'date_p'].map(lambda d: d.date() if d else None).values
    if 'dt_p' in T:
        O.loc[idx, 'dt'] = T.loc[idx, 'dt_p'].values
    biz.extra['orders_without_lines'] = len(new)


def _orders_only(biz, kinds, pidx):
    """Invoice list without line items: revenue of an order = goods total."""
    recs = []
    for t in kinds['orders']:
        prev = {}
        for r in t.data:
            oid = clean_str(t.col('order_id', r))
            if not oid:
                continue
            d = dict(order_id=oid)
            for c in ('date', 'datetime', 'status', 'cust_id', 'cust_name', 'note', 'channel', 'pay_method',
                      'staff_id'):
                d[c] = t.col(c, r)
            for c in ('goods_total', 'discount', 'order_disc', 'ship_fee', 'payable', 'paid', 'debt'):
                d[c] = _num(t.col(c, r))
            d['_year'] = t.year
            recs.append(d)
    O = pd.DataFrame(recs).drop_duplicates('order_id')
    O['dt'] = [parse_datetime(v, y) for v, y in zip(O['datetime'], O['_year'])]
    O['date'] = [parse_datetime(v, y) for v, y in zip(O['date'], O['_year'])]
    O['date'] = [d.date() if d else (x.date() if x else None) for d, x in zip(O['date'], O['dt'])]
    O['status_raw'] = O['status']
    O['status'] = O['status'].map(lambda s: status_of(s) or 'completed')
    for c in ('cust_id', 'cust_name', 'note', 'channel', 'pay_method'):
        O[c] = O[c].map(clean_str)
    O = O.rename(columns={'staff_id': 'staff'})
    O['order_disc'] = O['order_disc'].fillna(O['discount'])
    O['rev'] = O['goods_total']
    biz.orders = O.drop(columns=['datetime', '_year', 'discount']).reset_index(drop=True)
    biz.lines = None
    biz.products = pd.DataFrame(columns=['prod_key', 'prod_id', 'name', 'category', 'base_unit'])
    biz.prod_names = {}
    biz.prod_spellings = {}
    biz.pidx = pidx
    biz.extra['base_unit'] = {}
    _load_side_tables(biz, kinds, pidx)


def _split_toppings(L, pidx):
    """'Trà sữa (M) + 2 Trân châu đen (10.000)' -> item line + topping line."""
    out_rows = []
    topping_keys = {}
    for k, nm in pidx.names.items():
        if k.startswith('topping '):
            topping_keys[k[len('topping '):]] = k
    for k, d in pidx.info.items():
        if k and k.startswith('topping '):
            topping_keys[k[len('topping '):]] = k
    extra = []
    for i, name in L['prod_name'].items():
        if not isinstance(name, str) or ' + ' not in name:
            continue
        parts = name.split(' + ')
        base = parts[0].strip()
        top_total = 0.0
        for p in parts[1:]:
            m = re.match(r'^\s*(?:(\d+)\s+)?(.+?)\s*\(([^)]*)\)\s*$', p)
            if not m:
                continue
            n = float(m.group(1)) if m.group(1) else 1.0
            tname = m.group(2).strip()
            money = parse_number(m.group(3)) or 0.0
            tk = topping_keys.get(norm(tname), 'topping ' + norm(tname))
            top_total += money
            row = L.loc[i].copy()
            row['prod_name'] = 'Topping ' + tname
            row['prod_key'] = tk
            row['qty_n'] = n
            row['qty_base_n'] = n
            row['price_n'] = money / n if n else money
            row['amount_n'] = money
            row['unit_k'] = 'phan'
            row['cost_n'] = np.nan
            extra.append(row)
        L.at[i, 'prod_name'] = base
        L.at[i, 'prod_key'] = pidx.key_of(None, base)
        L.at[i, 'amount_n'] = L.at[i, 'amount_n'] - top_total
        pidx.names[pidx.key_of(None, base)][base] += 1
    if extra:
        L = pd.concat([L, pd.DataFrame(extra)], ignore_index=True)
    return L


def _load_side_tables(biz, kinds, pidx):
    from loader_side import load_side
    load_side(biz, kinds, pidx)
