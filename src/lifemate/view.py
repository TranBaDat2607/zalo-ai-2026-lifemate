"""The readings view (ASR_IMPROVEMENT_PLAN.md, Stage 1, 10 Nov 2026): the main transcript, with the other speech
models' readings shown in braces where they disagree with it.

    main    "Thứ mai bảy giờ nhắc anh uống canxi"
    others  "tối mai 7 giờ nhắc anh uống canxi", "thôi mai bảy giờ nhắc anh uống canxi", ...
    view    "{Thứ|tối|thôi} mai bảy giờ nhắc anh uống canxi"

Words are compared after asr_eval.normal_tokens (case, punctuation, digits spelled out, unified spellings) with
discourse particles ignored, so formatting never opens a brace. A reading that drops or adds words takes the
neighbouring word into the span, so every option is a real phrase. The main reading's option comes first, then the
others by how many readings gave them. Where every reading agrees, the view is the main text unchanged.

Why: on the 5,036 train clips (out-of-fold readings) the five speech models agree on 84 % of the words and are then
right 98 % of the time; where they disagree the main reading is right 62 % of the time. With one transcript the agent
cannot tell which words are shaky, and its restated request breaks as many key words as it fixes.
"""
import difflib
import random
import re
import unicodedata

from .asr_eval import normal_tokens

IGNORED = frozenset("nha nhé nhá nhe nghen nhen hen giùm dùm giúp hộ ạ à á ơi ừ ờ nhỉ nè luôn vậy đấy ấy".split())
MAX_SPANS, MAX_OPTIONS, MAX_WORDS = 6, 4, 6
MIN_RATIO = 0.5  # a reading sharing less than this with the main one is left out
_WORD = re.compile(r"[^\W_]+", re.U)
_TRAIL = re.compile(r"[^\w]+$", re.U)


def _units(text: str) -> tuple:
    """(surface words, [(compare token, index of its surface word)])."""
    words = text.split()
    return words, [(t, k) for k, w in enumerate(words) for t in normal_tokens(w) if t not in IGNORED]


def _bare(words: list) -> str:
    return " ".join(x for x in (w.strip(".,;:!?…\"'()[]{}|") for w in words) if x)


def spans_of(main: str, others: list) -> tuple:
    """(main surface words, [(first word, last word, [options], main support)]) with options = the main reading's span
    first, then the other readings' distinct spans by how many gave them; main support = how many other readings agree
    with the main reading there. Spans are whole surface words; overlapping and touching ones are merged."""
    words, units = _units(main)
    seq = [t for t, _ in units]
    n = len(seq)
    readings, ranges = [], []
    if not n:
        return words, []
    for text in others:
        if not text or not text.strip():
            continue  # a missing reading says nothing
        o_words, o_units = _units(text)
        o_seq = [t for t, _ in o_units]
        matcher = difflib.SequenceMatcher(a=seq, b=o_seq, autojunk=False)
        if not o_seq or matcher.ratio() < MIN_RATIO:
            continue  # particles only, or a different text altogether (a hallucination): one span over everything
        match = {}
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                match.update({i1 + d: j1 + d for d in range(i2 - i1)})
                continue
            if i1 == i2 or j1 == j2:  # added or dropped words: the neighbouring word comes into the span
                i1, i2 = (i1 - 1, i2) if i1 > 0 else (i1, i2 + 1)
            ranges.append((i1, min(i2, n)))
        readings.append((o_words, o_units, match))
    word_of = [k for _, k in units]
    first, last = {}, {}
    for idx, k in enumerate(word_of):
        first.setdefault(k, idx)
        last[k] = idx
    merged = []
    for a, b in sorted(ranges):
        a, b = first[word_of[a]], last[word_of[b - 1]] + 1  # whole surface words
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        main_key = " ".join(seq[a:b])
        counts, texts, support = {}, {}, 0
        for o_words, o_units, match in readings:
            # outside every span all readings match the main one, so both edges are anchored
            s = match[a - 1] + 1 if a > 0 else 0
            e = match[b] if b < n else len(o_units)
            key = " ".join(t for t, _ in o_units[s:e])
            if key == main_key:
                support += 1
                continue
            if s >= e:
                continue
            counts[key] = counts.get(key, 0) + 1
            texts.setdefault(key, _bare(o_words[o_units[s][1]:o_units[e - 1][1] + 1]))
        alts = sorted(counts, key=lambda k: -counts[k])  # stable: ties keep the readings' order
        out.append((word_of[a], word_of[b - 1], [_bare(words[word_of[a]:word_of[b - 1] + 1])] + [texts[k] for k in alts],
                    support))
    return words, out


def choose(main: str, others: list, max_spans: int = MAX_SPANS, max_options: int = MAX_OPTIONS,
           max_words: int = MAX_WORDS) -> tuple:
    """(main surface words, [(first word, last word, [options])] in order): the spans a view shows. At most
    ``max_spans`` spans (the most disputed ones), ``max_options`` options per span (the main one included) and
    ``max_words`` words per option; a span left with no other option is not shown."""
    words, spans = spans_of(main, others)
    kept = []
    for first, last, options, support in spans:
        alts = [o for o in options[1:] if o and len(o.split()) <= max_words][:max_options - 1]
        if alts and len(options[0].split()) <= max_words:
            kept.append((support, first, last, [options[0]] + alts))
    kept = sorted(sorted(kept)[:max_spans], key=lambda s: s[1])
    return words, [(first, last, options) for _, first, last, options in kept]


def build(main: str, others: list, **caps) -> str:
    """The view of one clip: ``main`` with ``{main option|other options}`` where the ``others`` disagree with it."""
    words, kept = choose(main, others, **caps)
    if not kept:
        return main
    at = {first: (last, options) for first, last, options in kept}
    out, k = [], 0
    while k < len(words):
        if k in at:
            last, options = at[k]
            trail = _TRAIL.search(words[last])
            out.append("{" + "|".join(options) + "}" + (trail.group() if trail else ""))
            k = last + 1
        else:
            out.append(words[k])
            k += 1
    return " ".join(out)


class ReadingsBank:
    """How the main reading and the other readings heard each said word together, on real clips, to give texts without
    audio (synthetic conversations) views with the real structure: the readings share most errors, and where they
    disagree the right word is often offered by none of them. Independent draws from a confusion table offer the right
    word in twice as many braces as real readings do (view_check.py).

    ``bank[word]`` lists the observed outcomes (one string per reading, '' = dropped, words a reading added are joined to
    the word before) other than "all readings heard it"; ``seen[word]`` counts the word's occurrences. Words never seen
    take the pattern of a random rare word (seen at most twice), with blurred spellings where that pattern was wrong."""

    def __init__(self, bank: dict, seen: dict, rare: list):
        self.bank, self.seen, self.rare = bank, seen, rare

    @classmethod
    def from_rows(cls, rows: list):
        """``rows``: (said text, [main reading, other readings...]) of real clips; every row has the same readings."""
        bank, seen = {}, {}
        for said, readings in rows:
            a = _said_words(said)
            outcomes = [_outcomes(a, _said_words(r or "")) for r in readings]
            for i, w in enumerate(a):
                seen[w] = seen.get(w, 0) + 1
                pattern = tuple(o[i] for o in outcomes)
                if any(x != w for x in pattern):
                    bank.setdefault(w, []).append(pattern)
        rare = [(w, p) for w, ps in bank.items() if seen[w] <= 2 for p in ps]
        rare_seen = sum(n for w, n in seen.items() if n <= 2)
        return cls(bank, seen, rare + [None] * max(0, rare_seen - len(rare)))

    def readings(self, text: str, rng: random.Random) -> tuple:
        """(main, [others]) for ``text``: each word gets the outcome pattern of a real occurrence of that word."""
        text = unicodedata.normalize("NFC", text)
        found = list(_WORD.finditer(text))
        n_readings = None
        patterns = []
        for m in found:
            w = m.group().lower()
            if w in self.seen:
                options = self.bank.get(w, [])
                pick = rng.randrange(self.seen[w])
                pattern = options[pick] if pick < len(options) else None
            else:
                drawn = rng.choice(self.rare) if self.rare else None
                pattern = None if drawn is None else _blurred(w, *drawn, rng)
            patterns.append(pattern)
            if pattern is not None:
                n_readings = len(pattern)
        n_readings = n_readings or 6

        def render(r: int) -> str:
            out, at = [], 0
            for m, pattern in zip(found, patterns):
                word = m.group()
                new = word if pattern is None else pattern[r]
                if new and new != word.lower() and word[0].isupper():
                    new = new[0].upper() + new[1:]
                elif new == word.lower():
                    new = word
                out.append(text[at:m.start()] + new)
                at = m.end()
            return " ".join(("".join(out) + text[at:]).split())

        return render(0), [render(r) for r in range(1, n_readings)]

    def view(self, text: str, rng: random.Random) -> str:
        return build(*self.readings(text, rng))


def _said_words(text: str) -> list:
    return _WORD.findall(unicodedata.normalize("NFC", text).lower())


def _outcomes(said: list, heard: list) -> list:
    """What one reading made of each said word: the word, a substitute, '' (dropped); added words go with the word
    before (or the first word)."""
    out = list(said)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=said, b=heard, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if op == "insert":
            k = i1 - 1 if i1 > 0 else 0
            if not said:
                continue
            out[k] = " ".join(x for x in ([out[k]] + heard[j1:j2] if i1 > 0 else heard[j1:j2] + [out[k]]) if x)
            continue
        span = list(range(i1, i2))
        got = heard[j1:j2]
        for n, i in enumerate(span):
            if n < len(got):
                out[i] = got[n] if n < len(span) - 1 else " ".join(got[n:])
            else:
                out[i] = ""
    return out


def _blurred(word: str, rare_word: str, pattern: tuple, rng: random.Random) -> tuple:
    """A never-seen word with a rare word's pattern: readings that heard that word right hear this one right, each
    distinct wrong outcome becomes one blurred spelling of this word."""
    from .noise import _blur
    spelled = {}
    out = []
    for x in pattern:
        if x == rare_word:
            out.append(word)
        elif x == "":
            out.append("")
        else:
            if x not in spelled:
                new, taken = x, set(spelled.values()) | {word}
                for _ in range(8 if len(word) > 2 else 0):  # distinct wrong outcomes stay distinct
                    new = _blur(word, rng)
                    if new not in taken:
                        break
                spelled[x] = new
            out.append(spelled[x])
    return tuple(out)
