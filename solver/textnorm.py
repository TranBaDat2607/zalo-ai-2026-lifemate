"""Text normalisation for Vietnamese questions and names.

norm() folds a string to lowercase ASCII: strips diacritics, repairs raw Telex
typing ("dduwowcj" -> "duoc"), and collapses punctuation to single spaces. The
same function is applied to questions and to names found in the data, so the
two sides always meet in the same space.
"""
import re
import unicodedata

_TELEX_FINAL = re.compile(r'^([a-z]*[aeiouy][a-z]*?)([sfrxj])$')
_VALID_FINAL = re.compile(r'(?:[aeiouy]|c|ch|m|n|ng|nh|p|t)$')
_ASCII_WORD = re.compile(r'[a-z]+')


def strip_accents(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s.replace('đ', 'd').replace('Đ', 'D')


def _fix_telex_word(w: str) -> str:
    if not _ASCII_WORD.fullmatch(w):
        return w
    orig = w
    w = re.sub(r'^dd', 'd', w)
    w = w.replace('uwow', 'uo').replace('uow', 'uo')
    w = re.sub(r'aa|aw', 'a', w)
    w = re.sub(r'ee', 'e', w)
    w = re.sub(r'oo|ow', 'o', w)
    w = w.replace('uw', 'u')
    m = _TELEX_FINAL.match(w)
    if m and len(w) > 2:
        stem = m.group(1)
        if _VALID_FINAL.search(stem):
            w = stem
    # tone letter typed in the middle, e.g. "nhieeuf" handled above; "thanhf"
    return w if w else orig


def norm(s) -> str:
    if s is None:
        return ''
    s = unicodedata.normalize('NFC', str(s)).lower()
    s = strip_accents(s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    words = [_fix_telex_word(w) for w in s.split()]
    return ' '.join(words)


def norm_keep(s) -> str:
    """Lowercase + accent strip, keep punctuation (for number/date parsing)."""
    if s is None:
        return ''
    s = unicodedata.normalize('NFC', str(s)).lower()
    return strip_accents(s)


def name_key(s) -> str:
    """Key for comparing entity names that differ only in spelling/case/spaces."""
    return norm(s)


def score_norm(s) -> str:
    """The competition's string normalisation (keeps diacritics)."""
    s = unicodedata.normalize('NFC', str(s)).lower()
    s = ''.join(c if (c.isalnum() or c.isspace()) else ' ' for c in s)
    return ' '.join(s.split())
