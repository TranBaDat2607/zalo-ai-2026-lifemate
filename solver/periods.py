"""Time expressions in questions -> date ranges (rulebook §2).

All functions work on the normalised question text (textnorm.norm) and the
as-of date. A Period is an inclusive [start, end] date range plus a kind tag
used to pick the comparison period.
"""
import calendar
import csv
import datetime as dt
import os
import re

D = dt.date


class Period:
    def __init__(self, start, end, kind, label='', **kw):
        self.start = start
        self.end = end
        self.kind = kind          # day, week, wtd, month, mtd, quarter, year, ytd, lastn, range, tet, asof
        self.label = label
        self.__dict__.update(kw)

    def __repr__(self):
        return f'Period({self.kind} {self.start}..{self.end})'

    def days(self):
        return (self.end - self.start).days + 1


def month_end(y, m):
    return D(y, m, calendar.monthrange(y, m)[1])


def add_months(y, m, k):
    m0 = y * 12 + (m - 1) + k
    return m0 // 12, m0 % 12 + 1


def month_period(y, m, today=None):
    return Period(D(y, m, 1), month_end(y, m), 'month', f'{y}-{m:02d}', y=y, m=m)


def resolve_month_no_year(m, today):
    y = today.year
    if D(y, m, 1) > today:
        y -= 1
    return y


def _year(y):
    y = int(y)
    return y + 2000 if y < 100 else y


# ---- calendar (lunar) -----------------------------------------------------------
_LUNAR = None


def load_calendar(path):
    global _LUNAR
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((D.fromisoformat(r['date']), int(r['lunar_day']), int(r['lunar_month']),
                         int(r['lunar_year']), r['lunar_leap'].lower() == 'true'))
    _LUNAR = rows


def lunar_to_solar(lyear, lmonth, lday, leap=False):
    for d, ld, lm, ly, lp in _LUNAR or []:
        if ly == lyear and lm == lmonth and ld == lday and lp == leap:
            return d
    return None


def tet_window(year):
    """Tết of lunar year `year`: 23 tháng Chạp (year-1) .. mùng 10 tháng Giêng (year)."""
    start = lunar_to_solar(year - 1, 12, 23)
    end = lunar_to_solar(year, 1, 10)
    return start, end


# ---- parsing ------------------------------------------------------------------------
MONTH_WORD = r'(?:thang|th|t)'
_DM_Y = re.compile(r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{4}|\d{2})\b')


def find_periods(q, today):
    """Return list of (pos, Period) found in normalised question q."""
    found = []

    def add(m, p):
        found.append((m.start(), m.end(), p))

    # explicit date ranges: tu 15 02 2026 den (het) 21 02 2026
    for m in re.finditer(r'\btu (?:ngay )?(\d{1,2}) (\d{1,2})(?: (\d{4}|\d{2}))? (?:den|toi|cho den|-)(?: het)?(?: ngay)? (\d{1,2}) (\d{1,2})(?: (\d{4}|\d{2}))?\b', q):
        d1, m1, y1, d2, m2, y2 = m.groups()
        y2 = _year(y2) if y2 else (today.year)
        y1 = _year(y1) if y1 else y2
        try:
            add(m, Period(D(y1, int(m1), int(d1)), D(y2, int(m2), int(d2)), 'range'))
        except ValueError:
            pass
    # month ranges: tu thang 2 2026 den thang 4 2026
    for m in re.finditer(r'\b(?:tu (?:thang ?|th ?|t)?|(?:thang ?|th ?|t))(\d{1,2}) (\d{4}|\d{2}) (?:den|toi|-)(?: het)?(?: thang ?| th ?| t| )(\d{1,2}) (\d{4}|\d{2})\b', q):
        m1, y1, m2, y2 = m.groups()
        a = D(_year(y1), int(m1), 1)
        b = month_end(_year(y2), int(m2))
        add(m, Period(a, b, 'monthrange', m1=(a.year, a.month), m2=(b.year, b.month)))
    # quarter
    for m in re.finditer(r'\b(?:quy|q)\s?([1-4])(?:\s+(?:nam\s+)?(\d{4}|\d{2}))?\b', q):
        qn = int(m.group(1))
        y = _year(m.group(2)) if m.group(2) else None
        if y is None:
            y = today.year
            if D(y, 3 * qn - 2, 1) > today:
                y -= 1
        a = D(y, 3 * qn - 2, 1)
        b = month_end(y, 3 * qn)
        add(m, Period(a, b, 'quarter', y=y, q=qn))
    # month with year: thang 7 2025 / th7 2025 / t7 2025 / 7 2025 / thang 7 nam 2025
    for m in re.finditer(r'\b(?:(?:thang|th|t)\s?)?(\d{1,2})(?:\s+nam)?\s+(\d{4}|\d{2})\b', q):
        mo, y = int(m.group(1)), _year(m.group(2))
        pre = q[max(0, m.start() - 6):m.start()]
        if not 1 <= mo <= 12 or not 2015 <= y <= 2035:
            continue
        worded = q[m.start()].isalpha()
        # skip 'dd mm yyyy' dates (handled below) and day-month pairs
        if not worded and re.search(r'\d\s$', pre) and not re.search(r'\d{4}\s$', pre):
            continue
        if not worded and re.search(r'\b(?:ngay|tu|den|het)\s$', pre):
            continue
        add(m, month_period(y, mo))
    # month lists sharing one year: 'th9 th10 th11 2025'
    for m in re.finditer(r'\b((?:(?:thang|th|t)\s?\d{1,2}\s+(?:va\s+)?)+)(?:thang|th|t)\s?\d{1,2}\s+(\d{4})\b', q):
        y = int(m.group(2))
        for mm in re.finditer(r'(?:thang|th|t)\s?(\d{1,2})', m.group(1)):
            mo = int(mm.group(1))
            if 1 <= mo <= 12:
                s = m.start() + mm.start()
                found.append((s, s + len(mm.group(0)), month_period(y, mo)))
    # full dates dd mm yyyy
    for m in re.finditer(r'\b(\d{1,2}) (\d{1,2}) (\d{4})\b', q):
        try:
            d = D(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            continue
        add(m, Period(d, d, 'day', date=d))
    # month without year: 'thang 3' (not followed by a year)
    for m in re.finditer(r'\b(?:thang|th)\s?(\d{1,2})\b(?!\s+(?:nam\s+)?\d)', q):
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            y = resolve_month_no_year(mo, today)
            add(m, month_period(y, mo))
    # tet
    for m in re.finditer(r'\btet(?: am lich| nguyen dan)?(?: nam)?(?: (\d{4})| (nay|ngoai|truoc|roi|trc|ni))?', q):
        y = int(m.group(1)) if m.group(1) else None
        back = 1 if m.group(2) in ('ngoai', 'truoc', 'roi', 'trc') else 0
        if y is None:
            # the Tết of this lunar year if already started, else last one
            y = today.year
            s, e = tet_window(y)
            if s and s > today:
                y -= 1
            y -= back
        s, e = tet_window(y)
        if s:
            add(m, Period(s, e, 'tet', y=y))
    for m in re.finditer(r'\b(?:thang|th|t)\s?(\d{1,2}) (?:cua )?nam (?:roi|ngoai|truoc|trc)\b', q):
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            add(m, month_period(today.year - 1, mo))
    # relative
    rel = [
        (r'\b(?:hom nay|hnay|bua nay|hom ni|bua ni|ngay hom nay|homnay|hum nay|ngay nay)\b', 'today'),
        (r'\b(?:hom qua|hqua|bua qua|ngay hom qua|homqua|hum qua)\b', 'yesterday'),
        (r'\b(?:tuan nay|tuan ni)\b', 'wtd'),
        (r'\b(?:tuan truoc|tuan roi|tuan trc|tuan vua roi|tuan qua)\b', 'lastweek'),
        (r'\b(?:thang nay|th nay|thang ni|th ni|t nay)\b', 'mtd'),
        (r'\b(?:thang truoc|thang roi|th truoc|th trc|thang trc|thang vua roi)\b', 'lastmonth'),
        (r'\b(?:tu dau nam|nam nay|nam ni)\b', 'ytd'),
        (r'\b(?:nam truoc|nam ngoai|nam roi)\b', 'lastyear'),
    ]
    for rx, kind in rel:
        for m in re.finditer(rx, q):
            add(m, relative(kind, today))
    for m in re.finditer(r'\b(\d{1,3}) ngay (?:gan nhat|gan day|vua qua|qua|cuoi|gan nhat tinh ca hom nay)\b', q):
        n = int(m.group(1))
        add(m, Period(today - dt.timedelta(days=n - 1), today, 'lastn', n=n))
    for m in re.finditer(r'\bnam (\d{4})\b', q):
        y = int(m.group(1))
        if not re.search(r'\d\s+(?:nam\s+)?$', q[:m.start() + 1]):
            add(m, Period(D(y, 1, 1), D(y, 12, 31), 'year', y=y))
    # day-month without year: ngay 19 3
    for m in re.finditer(r'\bngay (\d{1,2}) (\d{1,2})\b(?! \d)', q):
        try:
            d = D(today.year, int(m.group(2)), int(m.group(1)))
        except ValueError:
            continue
        if d > today:
            d = D(today.year - 1, d.month, d.day)
        add(m, Period(d, d, 'day', date=d))
    # remove periods contained in a larger match
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out = []
    for s, e, p in found:
        if any(s >= s2 and e <= e2 and (s, e) != (s2, e2) for s2, e2, _ in out):
            continue
        if any((s, e) == (s2, e2) for s2, e2, _ in out):
            continue
        out.append((s, e, p))
    return [(s, p) for s, e, p in out]


def relative(kind, today):
    if kind == 'today':
        return Period(today, today, 'day', date=today, rel=True)
    if kind == 'yesterday':
        d = today - dt.timedelta(days=1)
        return Period(d, d, 'day', date=d, rel=True)
    if kind == 'wtd':
        s = today - dt.timedelta(days=today.weekday())
        return Period(s, today, 'wtd')
    if kind == 'lastweek':
        s = today - dt.timedelta(days=today.weekday() + 7)
        return Period(s, s + dt.timedelta(days=6), 'week')
    if kind == 'mtd':
        return Period(D(today.year, today.month, 1), today, 'mtd', y=today.year, m=today.month)
    if kind == 'lastmonth':
        y, m = add_months(today.year, today.month, -1)
        return month_period(y, m)
    if kind == 'ytd':
        return Period(D(today.year, 1, 1), today, 'ytd', y=today.year)
    if kind == 'lastyear':
        return Period(D(today.year - 1, 1, 1), D(today.year - 1, 12, 31), 'year', y=today.year - 1)
    raise ValueError(kind)


def previous(p, today=None):
    """Comparison period ('kỳ trước') for p."""
    if p.kind == 'month':
        y, m = add_months(p.start.year, p.start.month, -1)
        return month_period(y, m)
    if p.kind == 'mtd':
        y, m = add_months(p.start.year, p.start.month, -1)
        last = month_end(y, m).day
        return Period(D(y, m, 1), D(y, m, min(p.end.day, last)), 'mtd_prev')
    if p.kind in ('week', 'wtd'):
        return Period(p.start - dt.timedelta(days=7), p.end - dt.timedelta(days=7), 'week')
    if p.kind == 'day':
        d = p.start - dt.timedelta(days=1)
        return Period(d, d, 'day', date=d)
    if p.kind == 'quarter':
        y, m = add_months(p.start.year, p.start.month, -3)
        return Period(D(y, m, 1), month_end(*add_months(y, m, 2)), 'quarter')
    if p.kind == 'year':
        return Period(D(p.start.year - 1, 1, 1), D(p.start.year - 1, 12, 31), 'year')
    if p.kind == 'ytd':
        return Period(D(p.start.year - 1, 1, 1), D(p.start.year - 1, p.end.month, min(p.end.day, 28 if p.end.month == 2 else p.end.day)), 'ytd_prev')
    n = p.days()
    return Period(p.start - dt.timedelta(days=n), p.start - dt.timedelta(days=1), 'range')
