"""Parsing of the noisy cell values found in the shop files.

Money and quantities come in many spellings: 84000, '84,000', '84.000đ',
'129 000', '105k', '2,76tr', '12tr460', '10,000.0000', '3 ly'. Dates come as
datetime objects, Excel serials, 'dd/mm/yyyy', '1 thg 3 2025', '1/3' (year
taken from the sheet), ISO strings.
"""
import datetime as dt
import functools
import math
import re

_EXCEL_EPOCH = dt.datetime(1899, 12, 30)

_RE_TR_MIX = re.compile(r'^(\d+)\s*tr\s*(\d{1,3})$')            # 12tr460
_RE_SUFFIX = re.compile(r'^([\d.,\s]+?)\s*(tr|trieu|triệu|k|ngan|ngàn|nghin|nghìn)$')
_RE_LEAD_NUM = re.compile(r'^([+-]?[\d.,\s]*\d)\s*([^\d\s].*)?$')


def _num_plain(s: str):
    """Number with thousands/decimal separators, VN or US style."""
    s = s.replace(' ', ' ').strip()
    s = re.sub(r'(?<=\d)\s+(?=\d{3}\b)', '', s)  # '129 000'
    s = s.replace(' ', '')
    if not s:
        return None
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        parts = s.split(',')
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and parts[0] not in ('', '0')):
            s = s.replace(',', '')
        else:
            s = s.replace(',', '.')
    elif '.' in s:
        parts = s.split('.')
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and parts[0] not in ('', '0')):
            s = s.replace('.', '')
    try:
        return float(s)
    except ValueError:
        return None


def _num_decimal_comma(s: str):
    s = s.replace(' ', '')
    if ',' in s and '.' in s:
        return _num_plain(s)
    s = s.replace(',', '.')
    if s.count('.') > 1:
        s = s.replace('.', '')
    try:
        return float(s)
    except ValueError:
        return None


def parse_number(v):
    """Return float or None."""
    try:
        return _parse_number_c(v)
    except TypeError:
        return _parse_number(v)


@functools.lru_cache(maxsize=500000)
def _parse_number_c(v):
    return _parse_number(v)


def _parse_number(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    s = str(v).strip().lower()
    if not s or s in ('nan', 'none', '-', '--', 'null', '#value!', '#n/a'):
        return None
    neg = False
    if s.startswith('(') and s.endswith(')'):
        neg, s = True, s[1:-1].strip()
    if s.startswith('-') or s.startswith('−'):
        neg, s = True, s[1:].strip()
    elif s.startswith('+'):
        s = s[1:].strip()
    for tok in ('vnđ', 'vnd', 'đồng', 'dong', 'đ', '₫'):
        if s.endswith(tok):
            s = s[: -len(tok)].strip()
    m = _RE_TR_MIX.match(s)
    if m:
        val = int(m.group(1)) * 1_000_000 + int(m.group(2).ljust(3, '0')) * 1_000
        return -val if neg else float(val)
    m = _RE_SUFFIX.match(s)
    if m:
        base = _num_decimal_comma(m.group(1))
        if base is None:
            return None
        mult = 1_000_000 if m.group(2) in ('tr', 'trieu', 'triệu') else 1_000
        val = round(base * mult, 6)
        return -val if neg else val
    val = _num_plain(s)
    if val is None:
        m = _RE_LEAD_NUM.match(s)  # '3 ly', '2 cái'
        if m:
            val = _num_plain(m.group(1))
    if val is None:
        return None
    return -val if neg else val


def parse_qty(v):
    """Quantity cell, possibly with unit text: returns (number, unit_text)."""
    try:
        return _parse_qty_c(v)
    except TypeError:
        return _parse_qty(v)


@functools.lru_cache(maxsize=200000)
def _parse_qty_c(v):
    return _parse_qty(v)


def _parse_qty(v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None, None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (None if (isinstance(v, float) and math.isnan(v)) else float(v)), None
    s = str(v).strip()
    m = re.match(r'^([+-]?\d+(?:[.,]\d+)?)\s*(.*)$', s)
    if m:
        num = _num_decimal_comma(m.group(1)) if ',' in m.group(1) else float(m.group(1))
        unit = m.group(2).strip() or None
        return num, unit
    return parse_number(s), None


_RE_DMY = re.compile(r'^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})(?:[ T,]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$')
_RE_YMD = re.compile(r'^(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})(?:[ T]+(\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?$')
_RE_THG = re.compile(r'^(\d{1,2})\s*(?:thg|tháng|thang)\s*(\d{1,2})[\s,/]*(\d{4})?$')
_RE_DM = re.compile(r'^(\d{1,2})[/.\-](\d{1,2})$')


def parse_datetime(v, default_year=None):
    """Return datetime.datetime or None."""
    try:
        return _parse_datetime_c(v, default_year)
    except TypeError:
        return _parse_datetime(v, default_year)


@functools.lru_cache(maxsize=500000)
def _parse_datetime_c(v, default_year):
    return _parse_datetime(v, default_year)


def _parse_datetime(v, default_year=None):
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        if 20000 < v < 80000:
            return _EXCEL_EPOCH + dt.timedelta(days=float(v))
        return None
    s = str(v).strip()
    if not s:
        return None
    m = _RE_YMD.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hh, mm, ss = int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0)
        return dt.datetime(y, mo, d, hh, mm, ss)
    m = _RE_DMY.match(s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        hh, mm, ss = int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0)
        return dt.datetime(y, mo, d, hh, mm, ss)
    m = _RE_THG.match(s.lower())
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else default_year
        if y:
            return dt.datetime(y, mo, d)
        return None
    m = _RE_DM.match(s)
    if m and default_year:
        return dt.datetime(default_year, int(m.group(2)), int(m.group(1)))
    if re.fullmatch(r'\d{5}(?:\.\d+)?', s):
        return _parse_datetime(float(s))
    return None


def parse_date(v, default_year=None):
    d = parse_datetime(v, default_year)
    return d.date() if d else None


def parse_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ('true', '1', 'x', 'có', 'co', 'yes', 'y', 'đúng', 'dung', 'kê đơn', 'ke don'):
        return True
    if s in ('false', '0', '', 'không', 'khong', 'no', 'n', 'sai'):
        return False
    return None


def clean_str(v):
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer():
            return str(int(v))
    s = str(v).replace(' ', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    return s or None
