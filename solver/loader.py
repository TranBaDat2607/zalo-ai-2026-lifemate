"""Load one shop's data package into a canonical set of pandas tables.

Canonical tables (all optional except orders):
  orders     one row per order: order_id, date, dt, status, cust_id, cust_name,
             cust_key, cust_group, staff, channel, pay_method, goods_total,
             order_disc, ship_fee, payable, paid, debt, note, table, guests, rev
  lines      one row per order line: order_id, prod_key, prod_name, unit, qty,
             qty_base, price, line_disc, amount, cost, is_fee
  products   prod_key, prod_id, name, category, base_unit, cur_price, cost,
             active, rx, ingredient
  payments   order_id, dt, amount, method
  purchases  rc_id, date, supplier, prod_key, unit, qty, qty_base, price,
             amount, rc_total, status, note
  stock      prod_key -> on-hand quantity at end of data (base unit)
  prices     prod_key, date, price
  ... plus family specific extras in Biz.extra
"""
import datetime as dt
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from readme import Readme
from schema import map_columns
from tables import read_sheets, read_sqlite, table_from_rows
from textnorm import norm
from values import clean_str, parse_bool, parse_datetime, parse_number, parse_qty

CACHE_DIR = os.environ.get('DATAMATE_CACHE', os.path.join(os.path.dirname(__file__), '..', '.cache'))

ORDER_FIELDS = ['date', 'datetime', 'time', 'status', 'cust_id', 'cust_name', 'cust_group', 'staff_id',
                'channel', 'pay_method', 'goods_total', 'order_disc', 'ship_fee', 'payable', 'paid',
                'debt', 'note', 'table', 'area', 'guests', 'phone', 'address', 'tradein', 'cust_paid_net',
                'discount_order']


class Biz:
    def __init__(self, bid, path):
        self.bid = bid
        self.path = path
        self.readme = None
        self.data_end = None
        self.orders = None
        self.lines = None
        self.products = None
        self.units = {}            # prod_key -> {unit_key: factor_to_base}
        self.customers = None
        self.payments = None
        self.purchases = None
        self.stock = None          # dict prod_key -> qty at data end
        self.stock_moves = None
        self.prices = None
        self.expenses = None
        self.deliveries = None
        self.due = None
        self.shifts = None
        self.lots = None
        self.line_lots = None
        self.returns = None
        self.extra = {}
        self.empty_cols = set()    # normalised names of documented-but-empty columns
        self.has_cust_id = False
        self.notes = []
        self.prod_names = {}       # prod_key -> display name

    # ---- helpers -------------------------------------------------------------
    def display_name(self, key):
        return self.prod_names.get(key, key)


# ----------------------------------------------------------------------------
# small helpers

def _year_hint(*names):
    for n in names:
        if not n:
            continue
        m = re.search(r'(20\d\d)', str(n))
        if m:
            return int(m.group(1))
    return None


def status_of(raw):
    s = norm(raw)
    if not s:
        return None
    if re.search(r'\bhuy\b|cancel|khong giao duoc|giao that bai|that bai', s):
        return 'cancelled'
    if re.fullmatch(r'khach tra|da tra|tra', s):
        return 'returned'
    if re.search(r'tra hang|doi tra|tra lai|return|hoan hang|hoan tra|khach tra lai|da tra ve|tra ve', s):
        return 'returned'
    if re.search(r'dang giao|shipping|dang xu ly|dang giao dich|dang van chuyen|dang chuyen|cho giao|delivering'
                 r'|dang gui|dang ship|cho lay', s):
        return 'shipping'
    return 'completed'


_RETAIL = {'khach le', 'le', 'kl', 'khach vang lai', 'vang lai', 'khach', 'k l', 'kh le', 'khachle', 'retail', 'walk in'}


def is_retail(name):
    if name is None:
        return True
    s = norm(name)
    return s == '' or s in _RETAIL


def cust_key(name):
    """Identity key of a customer written by name: spelling-insensitive, phone stripped."""
    if name is None:
        return None
    s = str(name)
    s = re.sub(r'[\(\[]?\s*(?:\+?84|0)?\d[\d .\-]{7,}\d\s*[\)\]]?', ' ', s)  # phone numbers
    s = norm(s)
    s = re.sub(r'\b(sdt|dt|tel)\b', ' ', s)
    s = ' '.join(s.split())
    return s or None


def unit_key(u):
    return norm(u) if u is not None else ''


def _num(v):
    x = parse_number(v)
    return np.nan if x is None else x


# ----------------------------------------------------------------------------
# reading

class RawTable:
    def __init__(self, fname, sheet, header, data, fmap):
        self.fname = fname
        self.sheet = sheet
        self.header = header
        self.data = data
        self.fmap = fmap
        self.year = _year_hint(sheet, fname)

    def has(self, *fields):
        return all(f in self.fmap for f in fields)

    def any(self, *fields):
        return any(f in self.fmap for f in fields)

    def col(self, field, r):
        j = self.fmap.get(field)
        if j is None:
            return None
        v = r[j]
        return None if v == '' else v

    def __repr__(self):
        return f'<{self.fname}[{self.sheet}] {len(self.data)} rows {sorted(self.fmap)}>'


def read_raw_tables(path, rd):
    out = []
    for f in sorted(os.listdir(path)):
        if f.lower().endswith('.md') or f.lower().endswith('.sqlite'):
            continue
        p = os.path.join(path, f)
        try:
            sheets = read_sheets(p)
        except Exception:
            continue
        fixed = _fix_mojibake(f)
        for sheet, rows in sheets:
            desc = rd.cols_for(fixed, sheet) or rd.cols_for(f, sheet)
            header, data = table_from_rows(rows, known=list(desc))
            if not header:
                continue
            fmap = map_columns(header, desc)
            out.append(RawTable(fixed, sheet, header, data, fmap))
    return out


def _fix_mojibake(name):
    """Zip archives extracted on Windows can garble UTF-8 names as cp437."""
    try:
        g = name.encode('cp437').decode('utf-8')
        return g
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


# ----------------------------------------------------------------------------
# units

def _parse_factor(v, units_of_prod):
    """Factor cell: number, or '15 vỉ' (in terms of another unit)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v), None
    n, u = parse_qty(v)
    return n, (unit_key(u) if u else None)


def resolve_units(raw):
    """raw: prod_key -> list of (unit_key, factor, in_unit_key or None). Returns prod -> {unit: factor}."""
    out = {}
    for pk, items in raw.items():
        fac = {}
        pending = list(items)
        for _ in range(5):
            rest = []
            for u, f, inu in pending:
                if f is None:
                    continue
                if inu is None or inu == '':
                    fac[u] = f
                elif inu in fac:
                    fac[u] = f * fac[inu]
                else:
                    rest.append((u, f, inu))
            pending = rest
            if not pending:
                break
        for u, f, inu in pending:  # unknown intermediate unit: treat as base
            fac.setdefault(u, f)
        out[pk] = fac
    return out


# ----------------------------------------------------------------------------
# main entry

def load_business(path, use_cache=True):
    path = os.path.abspath(path)
    bid = os.path.basename(path.rstrip('/\\'))
    stamp = max(os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path))
    import hashlib
    cache = os.path.join(CACHE_DIR, f'{bid}_{hashlib.md5(path.encode("utf-8")).hexdigest()[:10]}.pkl')
    if use_cache and os.path.exists(cache):
        try:
            with open(cache, 'rb') as fh:
                obj = pickle.load(fh)
            if obj.get('stamp') == stamp and obj.get('version') == LOADER_VERSION:
                return obj['biz']
        except Exception:
            pass
    biz = _load(path, bid)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache, 'wb') as fh:
            pickle.dump({'stamp': stamp, 'version': LOADER_VERSION, 'biz': biz}, fh)
    return biz


LOADER_VERSION = 10


def _load(path, bid):
    biz = Biz(bid, path)
    rd = Readme(open(os.path.join(path, 'README.md'), encoding='utf-8').read())
    biz.readme = rd
    biz.data_end = rd.data_end
    files = os.listdir(path)
    if any(f.endswith('.sqlite') for f in files):
        from loader_sqlite import load_sqlite
        load_sqlite(biz, os.path.join(path, [f for f in files if f.endswith('.sqlite')][0]))
    else:
        tables = read_raw_tables(path, rd)
        import loader_special as SP
        if SP.is_repair(tables):
            SP.load_repair(biz, tables)
        elif SP.is_phone_book(tables):
            SP.load_phone(biz, tables)
        else:
            from loader_flat import assemble
            assemble(biz, tables)
    finalize(biz)
    return biz


# ----------------------------------------------------------------------------
# finalisation shared by all families

def finalize(biz):
    o = biz.orders
    if o is None:
        raise RuntimeError(f'{biz.bid}: no orders')
    o['date'] = pd.to_datetime(o['date'], errors='coerce').dt.normalize()
    biz.extra['orders_no_date'] = int(o['date'].isna().sum())
    o = o[o['date'].notna()].copy()
    if 'dt' not in o:
        o['dt'] = pd.NaT
    o['dt'] = pd.to_datetime(o['dt'])
    for c in ['cust_id', 'cust_name', 'cust_group', 'staff', 'channel', 'pay_method', 'note']:
        if c not in o:
            o[c] = None
    for c in ['goods_total', 'order_disc', 'ship_fee', 'payable', 'paid', 'debt', 'rev']:
        if c not in o:
            o[c] = np.nan
    o['status'] = o['status'].fillna('completed')
    o['valid'] = ~o['status'].isin(['cancelled', 'returned'])
    # customer list: names and groups of coded customers
    c = biz.customers
    if c is not None and 'cust_id' in c and c['cust_id'].notna().any():
        cm = c.dropna(subset=['cust_id']).drop_duplicates('cust_id').set_index('cust_id')
        if 'name' in cm:
            o['cust_name'] = o['cust_name'].where(o['cust_name'].notna(), o['cust_id'].map(cm['name']))
        if 'group' in cm and cm['group'].notna().any():
            o['cust_group'] = o['cust_group'].where(o['cust_group'].notna(), o['cust_id'].map(cm['group']))
    # customer identity
    if o['cust_id'].notna().any():
        biz.has_cust_id = True
        o['cust_key'] = o['cust_id'].where(o['cust_id'].notna(), None)
    else:
        o['cust_key'] = o['cust_name'].map(lambda n: None if is_retail(n) else cust_key(n))
    o.loc[o['cust_name'].map(is_retail) & o['cust_id'].isna(), 'cust_key'] = None
    biz.orders = o.sort_values(['date', 'order_id']).reset_index(drop=True)
    if biz.lines is not None:
        l = biz.lines
        if 'is_fee' not in l:
            l['is_fee'] = False
        l['is_fee'] = l['is_fee'].fillna(False) | l['prod_key'].isin(['phi phuc vu'])
        if 'cost' not in l:
            l['cost'] = np.nan
        biz.lines = l
