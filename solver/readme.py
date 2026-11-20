"""Parsing of the per-shop README.md."""
import datetime as dt
import re
import unicodedata


class Readme:
    def __init__(self, text: str):
        self.text = unicodedata.normalize('NFC', text)
        self.columns = {}      # file (or 'file|sheet') -> {column: description}
        self.files = []
        self.units = []        # (product, big_unit, factor, base_unit)
        self.data_end = None
        self._parse()

    def _parse(self):
        cur = None
        in_units = False
        for line in self.text.splitlines():
            m = re.match(r'## Cột trong `(.+?)`(?:, sheet `(.+?)`)?', line)
            if m:
                cur = m.group(1) + ('|' + m.group(2) if m.group(2) else '')
                self.columns[cur] = {}
                in_units = False
                continue
            if line.startswith('## '):
                cur = None
            m = re.match(r'- `(.+?)` \((\w+)\)', line)
            if m and cur is None:
                self.files.append(m.group(1))
            m = re.match(r'- \*\*(.+?)\*\*: (.+)', line)
            if m and cur:
                self.columns[cur][m.group(1)] = m.group(2).strip()
            m = re.search(r'Dữ liệu có đến hết (\d{1,2})/(\d{1,2})/(\d{4})', line)
            if m:
                self.data_end = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            m = re.match(r'- \*\*[^*]+\*\*: (.+?) bán theo (\S+), 1 (\S+) = ([\d.,]+) (\S+?)[;.,]', line)
            if m:
                for prod in re.split(r',\s*', m.group(1)):
                    if prod.strip():
                        self.units.append((prod.strip(), m.group(3), float(m.group(4).replace(',', '.')), m.group(5)))
            if 'Quy đổi đơn vị' in line:
                in_units = True
                continue
            if in_units:
                m = re.match(r'\s+- (.+): 1 (.+?) = ([\d.,]+) (.+)$', line)
                if m:
                    self.units.append((m.group(1).strip(), m.group(2).strip(),
                                       float(m.group(3).replace(',', '.')), m.group(4).strip()))
                elif line.strip() and not line.startswith('  '):
                    in_units = False

    def cols_for(self, filename, sheet=None):
        """Column descriptions for a data file (matches templated names too)."""
        keys = []
        for k in self.columns:
            f, _, sh = k.partition('|')
            if sh and sheet and sh != sheet:
                continue
            if f == filename or _template_match(f, filename):
                keys.append(k)
        out = {}
        for k in keys:
            out.update(self.columns[k])
        return out

    def has(self, phrase):
        return phrase.lower() in self.text.lower()


def _template_match(pattern, filename):
    if '<' not in pattern:
        return False
    rx = re.escape(pattern)
    rx = re.sub(r'\\<[^>]*\\>|<[^>]*>', r'[^/]+', rx)
    return re.fullmatch(rx, filename) is not None
