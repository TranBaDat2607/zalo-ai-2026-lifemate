"""Score QUAN-Transcribe on every user turn of the train split: WER plus metrics for what the agent depends on.

    python -m lifemate.asr_eval --out outputs/asr_eval --limit 50

Only train.jsonl and the audio are read, never <split>_environment/.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
# QUAN is trained on one spelling of these; the choice doesn't change the meaning, so it shouldn't count as an error.
SPELLINGS = {"ngàn": "nghìn", "lẻ": "linh", "tỉ": "tỷ", "bẩy": "bảy"}
# Tone mark position differs between old and new orthography: "hoà" and "hòa" are the same syllable.
TONE_PAIRS = {"oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
              "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
              "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy"}
_ORTHO = re.compile("|".join(TONE_PAIRS))
_NUMBER = re.compile(r"\d+")
_WORD = re.compile(r"[^\W_]+", re.U)
_VI_SYLLABLE = re.compile(r"^(b|c|ch|d|g|gh|gi|h|k|kh|l|m|n|ng|ngh|nh|p|ph|q|qu|r|s|t|th|tr|v|x)?[aeiouy]+(c|ch|m|n|ng|nh|p|t)?$")


def _below_thousand(n: int, full: bool) -> list:
    hundreds, tens, units = n // 100, n // 10 % 10, n % 10
    out = []
    if hundreds or full:
        out += [DIGITS[hundreds], "trăm"]
    if tens == 0:
        if units and out:
            out.append("linh")
    elif tens == 1:
        out.append("mười")
    else:
        out += [DIGITS[tens], "mươi"]
    if units:
        if units == 5 and tens:
            out.append("lăm")
        elif units == 1 and tens > 1:
            out.append("mốt")
        elif units == 4 and tens > 1:
            out.append("tư")
        else:
            out.append(DIGITS[units])
    return out


def spell_number(text: str) -> list:
    """Vietnamese reading of a digit string. Leading zeros and long runs (phone numbers, IDs) are read digit by digit."""
    if len(text) >= 7 or (len(text) > 1 and text[0] == "0"):
        return [DIGITS[int(d)] for d in text]
    n = int(text)
    if n == 0:
        return ["không"]
    out = []
    if n >= 1000:
        out += _below_thousand(n // 1000, False) + ["nghìn"]
    if n % 1000:
        out += _below_thousand(n % 1000, n >= 1000)
    return out


def number_readings(text: str) -> list:
    """Accepted spoken forms: the standard reading, digit by digit, and the short form ("chín lăm" for 95)."""
    readings = [spell_number(text), [DIGITS[int(d)] for d in text]]
    if len(text) == 2 and text[1] != "0" and text[0] not in "01":
        readings.append([w for w in spell_number(text) if w != "mươi"])
    return readings


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return unicodedata.normalize("NFC", _ORTHO.sub(lambda m: TONE_PAIRS[m.group()], text))


def strict_tokens(text: str) -> list:
    """Case and punctuation removed, digits left as they are: the plain WER a raw comparison would give."""
    return _WORD.findall(_clean(text))


def normal_tokens(text: str) -> list:
    """Also spell out digits and unify spellings, because QUAN always writes numbers as words."""
    spaced = _NUMBER.sub(lambda m: " " + " ".join(spell_number(m.group())) + " ", _clean(text))
    return [SPELLINGS.get(w, w) for w in _WORD.findall(spaced)]


def edit_distance(ref: list, hyp: list) -> int:
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def _contains(tokens: list, needle: list) -> bool:
    return any(tokens[i:i + len(needle)] == needle for i in range(len(tokens) - len(needle) + 1))


def critical_terms(ref: str) -> dict:
    """The parts of a request that end up in tool arguments: numbers, names and foreign words."""
    numbers = _NUMBER.findall(ref)
    names, foreign = [], []
    first = True
    for raw in ref.split():
        word = raw.strip(".,?!:;…\"'()")
        ends_sentence = raw[-1] in ".?!…"
        if word and not any(c.isdigit() for c in word):
            for part in _WORD.findall(word):  # "S+" -> "S", which is too short to judge and is skipped below
                if len(part) < 2:
                    continue
                if (part.isascii() and part.isupper() and len(part) >= 2) or \
                        (part.isascii() and not _VI_SYLLABLE.match(part.lower())):
                    foreign.append(part)
                elif part[0].isupper() and not first:
                    names.append(part)
        first = ends_sentence
    return {"number": numbers, "name": names, "foreign": foreign}


def term_hits(ref: str, hyp: str) -> dict:
    """{kind: (found, total)}. A term counts when it appears anywhere in the transcript, so this is an upper bound."""
    tokens = normal_tokens(hyp)
    terms = critical_terms(ref)
    hits = {"number": (sum(any(_contains(tokens, [SPELLINGS.get(w, w) for w in r]) for r in number_readings(n))
                           for n in terms["number"]), len(terms["number"]))}
    available = Counter(tokens)
    for kind in ("name", "foreign"):
        found = 0
        for term in terms[kind]:
            word = SPELLINGS.get(_clean(term), _clean(term))
            if available[word] > 0:
                available[word] -= 1
                found += 1
        hits[kind] = (found, len(terms[kind]))
    return hits


def score_turn(ref: str, hyp: str) -> dict:
    strict_r, strict_h = strict_tokens(ref), strict_tokens(hyp)
    norm_r, norm_h = normal_tokens(ref), normal_tokens(hyp)
    chars_r, chars_h = "".join(norm_r), "".join(norm_h)
    hits = term_hits(ref, hyp)
    row = {"strict_errors": edit_distance(strict_r, strict_h), "strict_words": len(strict_r),
           "norm_errors": edit_distance(norm_r, norm_h), "norm_words": len(norm_r),
           "char_errors": edit_distance(list(chars_r), list(chars_h)), "chars": len(chars_r),
           "exact": norm_r == norm_h}
    for kind, (found, total) in hits.items():
        row[f"{kind}_found"], row[f"{kind}_total"] = found, total
    row["critical_ok"] = all(found == total for found, total in hits.values())
    row["has_critical"] = any(total for _, total in hits.values())
    return row


def summarise(rows: list) -> dict:
    total = lambda key: sum(r[key] for r in rows)  # noqa: E731
    ratio = lambda a, b: round(100 * a / b, 2) if b else None  # noqa: E731
    critical = [r for r in rows if r["has_critical"]]
    found = sum(total(f"{k}_found") for k in ("number", "name", "foreign"))
    terms = sum(total(f"{k}_total") for k in ("number", "name", "foreign"))
    return {
        "turns": len(rows),
        "wer_strict_pct": ratio(total("strict_errors"), total("strict_words")),
        "wer_normalised_pct": ratio(total("norm_errors"), total("norm_words")),
        "cer_normalised_pct": ratio(total("char_errors"), total("chars")),
        "exact_turn_pct": ratio(total("exact"), len(rows)),
        "critical_term_recall_pct": ratio(found, terms),
        "critical_terms": terms,
        "critical_turn_ok_pct": ratio(sum(r["critical_ok"] for r in critical), len(critical)),
        "critical_turns": len(critical),
        **{f"{k}_recall_pct": ratio(total(f"{k}_found"), total(f"{k}_total")) for k in ("number", "name", "foreign")},
        **{f"{k}_terms": total(f"{k}_total") for k in ("number", "name", "foreign")},
    }


def turns_of(items: list) -> list:
    """Every user turn, tagged with its sample: history turns are transcribed at inference too."""
    turns = []
    for item in items:
        for t in item["history"]:
            if t["role"] == "user" and t.get("audio"):
                turns.append({"id": item["id"], "kind": "history", "audio": t["audio"], "ref": t["audio_text"]})
        turns.append({"id": item["id"], "kind": "latest", "audio": item["audio"], "ref": item["audio_text"]})
    return turns


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"no HF_TOKEN Kaggle secret: {type(e).__name__}", flush=True)
        return None


def run_model(asr, turns: list, audio_root: Path, use_lm: bool) -> None:
    """One file at a time, the way inference sees them, so the timings are real per-file latencies."""
    key = "lm" if use_lm else "greedy"
    for i, t in enumerate(turns):
        start = time.monotonic()
        t[f"hyp_{key}"] = asr.transcribe(str(audio_root / t["audio"]), use_lm=use_lm)
        t[f"seconds_{key}"] = time.monotonic() - start
        if i % 500 == 0:
            print(f"{key}: {i}/{len(turns)}", flush=True)


def report(turns: list, key: str) -> dict:
    rows = [{**t, **score_turn(t["ref"], t[f"hyp_{key}"])} for t in turns]
    latest = [r for r in rows if r["kind"] == "latest"]
    by_sample = Counter()
    for t in turns:
        by_sample[t["id"]] += t[f"seconds_{key}"]
    seconds = sorted(t[f"seconds_{key}"] for t in turns[1:] or turns)  # the first call includes warm-up
    return {
        "all_turns": summarise(rows),
        "latest_turn": summarise(latest),
        "seconds_per_turn_mean": round(sum(seconds) / len(seconds), 4),
        "seconds_per_turn_p99": round(seconds[int(len(seconds) * 0.99)], 4),
        "seconds_per_sample_max": round(max(by_sample.values()), 3),
    }, rows


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/asr_eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--model", default="qnguyen3/QUAN-Transcribe-1.0")
    args = ap.parse_args(argv)

    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from quan_transcribe import QuanTranscribe
    from transformers import AutoConfig

    from . import data

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    root = data.find_root("train")
    items = data.load_split("train", root)
    if args.limit:
        items = items[: args.limit]
    turns = turns_of(items)
    audio_root = Path(root) / "train"
    missing = [t["audio"] for t in turns if not (audio_root / t["audio"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} audio files missing, first: {missing[0]}")
    print(f"{len(items)} samples, {len(turns)} turns", flush=True)
    print("base encoder config:", AutoConfig.from_pretrained("google/embeddinggemma-2").model_type, flush=True)

    # The private weights dataset needs no Hugging Face credential on Kaggle.
    local = next(iter(data.walk_input("model.safetensors")), None)
    if local:
        path = str(local)
    else:
        path = snapshot_download(args.model, token=hf_token(),
                                 allow_patterns=["config.json", "units.json", "model.safetensors", "lm/*"])

    asr = QuanTranscribe.from_pretrained(path)
    info = {"model": args.model, "revision": Path(path).name, "device": str(asr.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "parameters": sum(p.numel() for p in asr.model.parameters())}
    durations = [sf.info(str(audio_root / t["audio"])).duration for t in turns]
    info["audio_seconds"] = round(float(np.sum(durations)), 1)

    start = time.monotonic()
    run_model(asr, turns, audio_root, use_lm=False)
    info["greedy_realtime_factor"] = round(info["audio_seconds"] / (time.monotonic() - start), 1)
    if torch.cuda.is_available():
        info["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    results, row_sets = {}, {}
    results["greedy"], row_sets["greedy"] = report(turns, "greedy")

    if not args.no_lm:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kenlm"], check=True)
            asr.load_lm(os.path.join(path, "lm"))
            run_model(asr, turns, audio_root, use_lm=True)
            results["lm"], row_sets["lm"] = report(turns, "lm")
        except Exception as e:  # greedy is the main result, so an LM problem must not lose it
            info["lm_error"] = f"{type(e).__name__}: {e}"
            print("LM skipped:", info["lm_error"], flush=True)

    metrics = {**info, "results": results}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    for key, rows in row_sets.items():
        fields = ["id", "kind", "audio", "ref", f"hyp_{key}", f"seconds_{key}", "strict_errors", "strict_words",
                  "norm_errors", "norm_words", "char_errors", "chars", "exact", "number_found", "number_total",
                  "name_found", "name_total", "foreign_found", "foreign_total", "critical_ok", "has_critical"]
        with open(out / f"predictions_{key}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(metrics, indent=1, ensure_ascii=False), flush=True)
    return metrics


if __name__ == "__main__":
    main()
