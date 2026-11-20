"""Make clean text look like ASR output, using the word confusions of real transcripts.

Synthetic conversations have no audio, and LLM training can start before the fine-tuned ASR exists. Either way the LLM should
see the kind of text it gets at inference, so requests get the substitutions a real ASR made on train audio. Case and punctuation
stay as they are, because a fine-tuned Whisper keeps them.
"""
import difflib
import json
import random
import re
import unicodedata
from pathlib import Path

_WORD = re.compile(r"[^\W_]+", re.U)
PACKAGED = Path(__file__).parent / "confusions.json"


def _words(text: str) -> list:
    return _WORD.findall(unicodedata.normalize("NFC", text).lower())


class ConfusionNoise:
    def __init__(self, table: dict, seen: dict, wer: float, scale: float = 1.0):
        self.table, self.seen, self.wer, self.scale = table, seen, wer, scale
        self.unknown_rate = wer * scale  # words never heard in train get the average error rate

    @classmethod
    def from_pairs(cls, pairs: list, scale: float = 1.0):
        """``pairs`` holds (reference, hypothesis) transcripts."""
        table, seen, errors, total = {}, {}, 0, 0
        for ref, hyp in pairs:
            a, b = _words(ref), _words(hyp)
            total += len(a)
            for w in a:
                seen[w] = seen.get(w, 0) + 1
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
                if op == "equal":
                    continue
                errors += max(i2 - i1, j2 - j1)
                if op == "replace":
                    for x, y in zip(a[i1:i2], b[j1:j2]):
                        table.setdefault(x, []).append(y)
                elif op == "delete":
                    for x in a[i1:i2]:
                        table.setdefault(x, []).append("")
        return cls(table, seen, errors / max(total, 1), scale)

    @classmethod
    def packaged(cls, scale: float = 1.0):
        data = json.loads(PACKAGED.read_text(encoding="utf-8"))
        return cls(data["table"], data["seen"], data["wer"], scale)

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps({"wer": self.wer, "table": self.table, "seen": self.seen}, ensure_ascii=False, indent=0), encoding="utf-8")

    def _word(self, word: str, rng: random.Random):
        low = word.lower()
        options = self.table.get(low)
        if options:
            rate = self.scale * len(options) / max(self.seen.get(low, len(options)), len(options))
            if rng.random() >= rate:
                return word
            choice = rng.choice(options)
        elif low not in self.seen and len(low) > 2 and rng.random() < self.unknown_rate:
            choice = _blur(low, rng)
        else:
            return word
        return choice.capitalize() if choice and word[0].isupper() else choice

    def word(self, word: str, rng: random.Random) -> str:
        """One word as the ASR might have heard it ('' when it was dropped)."""
        return self._word(word, rng)

    def apply(self, text: str, rng: random.Random) -> str:
        text = unicodedata.normalize("NFC", text)
        noisy = _WORD.sub(lambda m: self._word(m.group(), rng), text)
        return " ".join(noisy.split())


def _blur(word: str, rng: random.Random) -> str:
    """An unseen word: lose its tone mark or one letter, the commonest shape of an error on a name."""
    if rng.random() < 0.6:
        stripped = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
        return unicodedata.normalize("NFC", stripped)
    i = rng.randrange(len(word))
    return word[:i] + word[i + 1:]
