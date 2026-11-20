"""Scorer implementing the competition matching rules (used on the train split)."""
from textnorm import score_norm


def _num_eq(a, b):
    try:
        return abs(round(float(a), 2) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def _str_eq(a, forms):
    if not isinstance(a, str):
        return False
    na = score_norm(a)
    return any(na == score_norm(f) for f in forms if isinstance(f, str))


def _elem_eq(a, ref, aliases):
    if isinstance(ref, (int, float)) and not isinstance(ref, bool):
        return _num_eq(a, ref)
    forms = [ref] + list(aliases or [])
    return _str_eq(a, forms)


def score(pred, ref):
    """+1 correct, 0 abstain, -1 wrong."""
    if pred is None or pred.get('type') in (None, 'abstain'):
        return 0
    rt = ref['type']
    pt = pred['type']
    if rt == 'unanswerable':
        return 1 if pt == 'unanswerable' else -1
    if pt != rt:
        return -1
    a, r = pred.get('answer'), ref.get('answer')
    alias = ref.get('alias') or []
    ok = False
    if rt in ('number', 'percent'):
        ok = isinstance(a, (int, float)) and not isinstance(a, bool) and _num_eq(a, r)
    elif rt == 'date':
        ok = a == r
    elif rt == 'boolean':
        ok = a is r or a == r and isinstance(a, bool)
    elif rt == 'string':
        ok = _str_eq(a, [r] + list(alias))
    elif rt == 'list':
        if isinstance(a, list) and len(a) == len(r):
            ok = all(_elem_eq(x, y, alias[i] if i < len(alias) else []) for i, (x, y) in enumerate(zip(a, r)))
    elif rt == 'table':
        ok = _table_eq(a, r, alias)
    return 1 if ok else -1


def _table_eq(a, r, alias):
    if not isinstance(a, dict) or 'rows' not in a:
        return False
    ra, rr = a.get('rows') or [], r.get('rows') or []
    if len(ra) != len(rr):
        return False
    ca = [score_norm(c) for c in a.get('columns', [])]
    cr = [score_norm(c) for c in r.get('columns', [])]
    if len(ca) != len(cr):
        return False
    if sorted(ca) == sorted(cr):
        perm = [ca.index(c) for c in cr]
    else:
        perm = list(range(len(cr)))
    used = [False] * len(ra)
    for i, row in enumerate(rr):
        al = alias[i] if i < len(alias) else []
        hit = False
        for j, cand in enumerate(ra):
            if used[j] or len(cand) != len(row):
                continue
            cand = [cand[k] for k in perm]
            if all(_elem_eq(x, y, al if isinstance(y, str) else []) for x, y in zip(cand, row)):
                used[j] = hit = True
                break
        if not hit:
            return False
    return True
