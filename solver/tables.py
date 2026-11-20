"""Reading raw csv / xlsx / sqlite files into header + rows."""
import csv
import os
import re
import sqlite3
import unicodedata

import python_calamine as pc

from textnorm import norm

_FOOTER_ACC = re.compile(r'^\s*(tổng|cộng)\b', re.I)
_FOOTER_ASCII = re.compile(r'^(tong cong|tong (ngay|thang|nam|quy|so|tien|ky)|cong (ngay|thang|nam|quy|ky)|ghi chu)\b')


def read_sheets(path):
    """Return list of (sheet_name, rows); rows are lists of python values ('' for empty)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        with open(path, encoding='utf-8-sig', newline='') as f:
            rows = [list(r) for r in csv.reader(f)]
        return [(None, rows)]
    if ext in ('.xlsx', '.xlsm', '.xls'):
        wb = pc.CalamineWorkbook.from_path(path)
        out = []
        for name in wb.sheet_names:
            try:
                rows = wb.get_sheet_by_name(name).to_python(skip_empty_area=False)
            except Exception:
                rows = []
            out.append((name, [list(r) for r in rows]))
        return out
    raise ValueError(f'unsupported file {path}')


def read_sqlite(path):
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    out = {}
    for (t,) in con.execute("select name from sqlite_master where type='table'"):
        cur = con.execute(f'select * from "{t}"')
        cols = [d[0] for d in cur.description]
        out[t] = (cols, [list(r) for r in cur.fetchall()])
    con.close()
    return out


def _nonempty(r):
    return [c for c in r if c not in (None, '')]


_NOTE_COLS = re.compile(r'(ghi chu|note|luu y|dien giai|mo ta|ly do|nguyen nhan)')


def is_footer_row(row, header=None, skip=()):
    if header is not None and not skip:
        skip = {j for j, h in enumerate(header) if _NOTE_COLS.search(norm(h))}
    for j, c in enumerate(row):
        if j in skip:
            continue
        if isinstance(c, str) and c and c.lstrip()[:1] in 'TtCcGg':
            s = unicodedata.normalize('NFC', c.strip())
            if not _FOOTER_QUICK.match(s):
                continue
            if _FOOTER_ACC.match(s) or _FOOTER_ASCII.match(norm(s)):
                return True
    return False


_FOOTER_QUICK = re.compile(r'^(tổng|tong|cộng|cong|ghi ch)', re.I)


def find_header(rows, known=None, scan=12):
    """Index of the header row. `known` = iterable of expected column names."""
    known_n = {norm(k) for k in (known or [])}
    best, best_score = 0, -1
    for i, r in enumerate(rows[:scan]):
        cells = [c for c in r if isinstance(c, str) and c.strip()]
        if len(cells) < 2:
            continue
        score = len(cells)
        if known_n:
            score = sum(1 for c in cells if norm(c) in known_n) * 100 + len(cells)
        # a header row has no numbers / dates
        if any(not isinstance(c, str) and c not in (None, '') for c in r):
            score -= 50
        if score > best_score:
            best, best_score = i, score
    return best


def table_from_rows(rows, known=None):
    """Return (header, data_rows). Drops pre-header lines, blank rows, footers."""
    if not rows:
        return [], []
    hi = find_header(rows, known)
    header = [str(c).strip() if c not in (None, '') else '' for c in rows[hi]]
    width = len(header)
    data = []
    skip = {j for j, h in enumerate(header) if _NOTE_COLS.search(norm(h))} or {-1}
    for r in rows[hi + 1:]:
        r = list(r) + [''] * (width - len(r))
        r = r[:width]
        ne = _nonempty(r)
        if not ne:
            continue
        if [str(c).strip() for c in r] == header:
            continue  # repeated header
        if is_footer_row(r, None, skip):
            continue
        data.append(r)
    return header, data


def records(header, data):
    return [dict(zip(header, r)) for r in data]
