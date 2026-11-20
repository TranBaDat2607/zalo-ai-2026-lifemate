"""Write worked notes v2 with Gemini, not Claude: the same branch-blind inputs, guide and checker as the v2 plan.

    python scripts/notes/gemini_notes.py write notes_v2/inputs/pilot.txt --name gm_pilot --budget-usd 0.5
    python scripts/notes/gemini_notes.py write 'notes_v2/gemini/inputs/batch_*_p*.txt' --jobs 4
    python scripts/notes/gemini_notes.py fix 'notes_v2/gemini/parts/gm_*.jsonl'   # one fix round for failing notes
    python scripts/notes/gemini_notes.py cost

Run with LIFEMATE_DATA set (the checker loads train). The API key comes from GEMINI_API_KEY or
~/.config/gemini/api_key, never from the repo. The system instruction is notes_v2/GUIDE.md with its input and output
paragraphs reworded for an API call; the rules and the made-up examples are unchanged. Every request is checked for
validation ids before it is sent. Each call's tokens and cost go to notes_v2/gemini/usage.jsonl, and no new call starts
once the logged spend reaches --budget-usd. Outputs: notes_v2/gemini/parts/<name>.jsonl (first pass) and
<name>_fix<k>.jsonl (fix rounds); check results in notes_v2/gemini/checks/. Claude writes and edits no note text: notes
are only normalised to one line, failures go back to Gemini with the checker's messages, or are dropped.
"""
import argparse
import glob
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTES2 = REPO / "notes_v2"
GEM = NOTES2 / "gemini"
PARTS, CHECKS = GEM / "parts", GEM / "checks"
USAGE, SOURCES = GEM / "usage.jsonl", GEM / "parts" / "sources.json"
PRICES = {  # USD per 1M tokens, paid tier, standard; thinking billed as output (Google AI pricing page, 8 Nov 2026)
    "gemini-3.8-flash": (0.75, 3.75), "gemini-3.6-flash": (0.75, 3.75), "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash": (0.30, 2.50), "gemini-3.1-pro-preview": (2.00, 12.00),
}
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
HEAD = re.compile(r"^=== (\S+) ===$", re.M)
LOCK = threading.Lock()
SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "required": ["id", "note"],
                                     "properties": {"id": {"type": "STRING"}, "note": {"type": "STRING"}}}}


def api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or (Path.home() / ".config" / "gemini" / "api_key").read_text()).strip()


def validation_ids() -> set:
    s = json.loads((REPO / "src" / "lifemate" / "splits.json").read_text())
    ids = set(s["random"]) | set(s["apps"])
    assert len(ids) == 386, len(ids)
    return ids


def system_text() -> str:
    g = (NOTES2 / "GUIDE.md").read_text(encoding="utf-8")
    g, n = re.subn(r"Read only the input file named in your task.*?Each sample\nblock has these fields:",
                   "The samples are in the user message, one block per sample, each starting with `=== train_XXXX ===`. "
                   "Each sample block has these fields:", g, flags=re.S)
    assert n == 1, "GUIDE.md input paragraph changed"
    g = g[:g.index("## Output")] + (
        "## Output\n\nReturn only a JSON array with one object per sample, in input order: "
        '`{"id": "train_XXXX", "note": "..."}`. The note is one line of plain text. When asked to fix notes, return '
        "objects only for the samples listed.\n")
    return g


def blocks(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    heads = list(HEAD.finditer(text))
    return {m.group(1): text[m.start():(heads[k + 1].start() if k + 1 < len(heads) else len(text))].strip()
            for k, m in enumerate(heads)}


def read_jsonl(path: Path) -> list:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def spent() -> float:
    return sum(r["usd"] for r in read_jsonl(USAGE))


def call(args, user: str, task: str) -> list:
    with LOCK:
        if spent() >= args.budget_usd:
            raise RuntimeError(f"budget reached ({spent():.3f} USD logged, cap {args.budget_usd})")
    bad = [v for v in VALIDATION if v in user]
    assert not bad, f"validation ids in a request: {bad[:3]}"
    cfg = {"responseMimeType": "application/json", "responseSchema": SCHEMA, "maxOutputTokens": 32768}
    if args.thinking:
        cfg["thinkingConfig"] = {"thinkingLevel": args.thinking}
    data = json.dumps({"systemInstruction": {"parts": [{"text": SYSTEM}]},
                       "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": cfg}).encode()
    for attempt in range(6):
        t = time.time()
        try:
            req = urllib.request.Request(URL.format(model=args.model), data=data,
                                         headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(10 * 2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 5:
                raise
            time.sleep(10 * 2 ** attempt)
    u = r.get("usageMetadata", {})
    pin, pout = PRICES[args.model]
    usd = (u.get("promptTokenCount", 0) * pin + (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)) * pout) / 1e6
    with LOCK, USAGE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": r.get("modelVersion", args.model),
                            "task": task, "thinking": args.thinking, "prompt": u.get("promptTokenCount", 0),
                            "output": u.get("candidatesTokenCount", 0), "thoughts": u.get("thoughtsTokenCount", 0),
                            "usd": round(usd, 5), "seconds": round(time.time() - t, 1)}) + "\n")
    cand = (r.get("candidates") or [{}])[0]
    if cand.get("finishReason") not in (None, "STOP"):
        raise RuntimeError(f"{task}: finish reason {cand.get('finishReason')}")
    return json.loads("".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought")))


def ask(args, todo: list, text_of, task: str, lead: str) -> dict:
    """Ask Gemini for notes on `todo` ids, per_call at a time; one retry for ids that do not come back."""
    notes = {}
    for _ in range(2):
        rest = [i for i in todo if i not in notes]
        for k in range(0, len(rest), args.per_call):
            chunk = rest[k:k + args.per_call]
            try:
                got = call(args, lead.format(n=len(chunk)) + "\n\n" + "\n\n".join(text_of(i) for i in chunk), f"{task}:{k}")
            except (RuntimeError, json.JSONDecodeError) as e:
                print(f"  {task}: {e}", flush=True)
                if "budget reached" in str(e):
                    return notes
                continue
            for rec in got if isinstance(got, list) else []:
                if isinstance(rec, dict) and rec.get("id") in chunk and isinstance(rec.get("note"), str):
                    notes.setdefault(rec["id"], " ".join(rec["note"].split()))
    return notes


REDO_LEAD = (
    " These are rewrites: the earlier notes left out sub-tasks. The request may have several sub-tasks; name the tools of "
    "EVERY sub-task in the request's order, including the lookups that supply an ID or a value (for example contacts.search "
    "before a top-up, call or transfer to a named person, banking.list_beneficiaries before banking.transfer, "
    "food_delivery.list_restaurants before get_menu and place_order, a balance read when the request asks for it). Tie "
    "each 'khớp đúng một' rule to the lookup that actually finds that item (a phone number or contact_id comes from "
    "contacts.search, not from a bill list). After the matching branch, continue with the remaining sub-tasks "
    "('rồi ...'). Do not invent a condition the request does not state; never write 'thì thôi, không ... thì thôi'. "
    "All installed tools are listed for every sample; choose from them.")


def batch_no(name: str):
    m = re.search(r"batch_(\d{3})", name)
    return int(m.group(1)) if m else None


def check(path: Path) -> dict:
    import check_notes_v2 as c2
    with LOCK:
        res = c2.check_file(path, write=False)
    CHECKS.mkdir(parents=True, exist_ok=True)
    (CHECKS / f"{path.stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False), flush=True)
    for sid, r in res["notes"].items():
        if not r["ok"]:
            print(f"  FAIL {sid}: {'; '.join(r['problems'])}", flush=True)
    return res


def write_one(args, src: Path) -> None:
    name = args.name or f"gm_{src.stem}"
    out = PARTS / f"{name}.jsonl"
    if out.exists() and not args.force:
        print(f"skip {name}: exists", flush=True)
        return
    bl = blocks(src)
    notes = ask(args, list(bl), bl.get, name, "Write the worked note for each of these {n} samples." + (REDO_LEAD if "_r2" in name else ""))
    out.write_text("".join(json.dumps({"id": i, "note": notes[i]}, ensure_ascii=False) + "\n" for i in bl if i in notes),
                   encoding="utf-8")
    with LOCK:
        src_map = json.loads(SOURCES.read_text()) if SOURCES.exists() else {}
        src_map[name] = str(src.relative_to(REPO))
        SOURCES.write_text(json.dumps(src_map, indent=0))
    check(out)


def fix_one(args, first: Path) -> None:
    import check_notes_v2 as c2
    base = re.sub(r"_fix\d+$", "", first.stem)
    src = REPO / json.loads(SOURCES.read_text())[base]
    bl = blocks(src)
    latest = {}
    fixes = sorted(PARTS.glob(f"{base}_fix*.jsonl"), key=lambda p: int(re.search(r"_fix(\d+)$", p.stem).group(1)))
    for p in [PARTS / f"{base}.jsonl"] + fixes:
        for r in read_jsonl(p):
            latest[r["id"]] = r["note"]
    with LOCK:
        fails = {i: (c2.check_note_v2(i, latest[i], batch_no(base)) if i in latest else ["no note returned"]) for i in bl}
    fails = {i: p for i, p in fails.items() if p}
    if not fails:
        return
    k = len(fixes) + 1
    if k > args.rounds:
        print(f"{base}: {len(fails)} still failing after {args.rounds} fix rounds", flush=True)
        return
    text_of = lambda i: f"{bl[i]}\nprevious note: {latest.get(i, '(none)')}\nchecker problems: {'; '.join(fails[i])}"
    notes = ask(args, list(fails), text_of, f"{base}_fix{k}",
                "These {n} notes failed the checker. For each sample, write a new note that follows the guide and fixes "
                "every listed problem." + (REDO_LEAD if "_r2" in base else ""))
    out = PARTS / f"{base}_fix{k}.jsonl"
    out.write_text("".join(json.dumps({"id": i, "note": notes[i]}, ensure_ascii=False) + "\n" for i in fails if i in notes),
                   encoding="utf-8")
    check(out)


def assemble() -> None:
    """notes_v2/gemini/batch_NNN.jsonl and src/lifemate/notes_v2.jsonl (what `train --notes package-v2` reads): per sample
    the last Gemini note that passes the checker (fix rounds after the first pass). A sample without a passing note is
    dropped and listed with its problems in notes_v2/gemini/checks/dropped.json. Only Gemini parts are read."""
    import check_notes_v2 as c2
    order = json.loads((GEM / "inputs" / "order.json").read_text())
    validation = validation_ids()
    fix_no = lambda p: int(m.group(1)) if (m := re.search(r"_fix(\d+)$", p.stem)) else 0
    final, dropped, first_ok, words, tokens = {}, {}, 0, [], []
    drop_file = CHECKS / "review2_drop.json"
    REVIEW_DROP = json.loads(drop_file.read_text()) if drop_file.exists() else {}
    for b, ids in enumerate(order):
        tries = {}
        for p in sorted(PARTS.glob(f"gm_batch_{b:03d}_p*.jsonl"), key=fix_no):
            for r in read_jsonl(p):
                tries.setdefault(r["id"], []).append(r["note"])
        redo = {}
        for p in sorted(PARTS.glob(f"gm_batch_{b:03d}_r2*.jsonl"), key=fix_no):
            for r in read_jsonl(p):
                redo.setdefault(r["id"], []).append(r["note"])
        tries.update(redo)  # second pass (8 Nov review): only its notes count for the ids it rewrote
        assert set(tries) <= set(ids), f"batch {b}: ids outside the batch"
        keep = []
        for sid in ids:
            notes = tries.get(sid, [])
            probs = [c2.check_note_v2(sid, n, b) for n in notes]
            first_ok += bool(probs) and not probs[0]
            good = [n for n, pr in zip(notes, probs) if not pr]
            if sid in REVIEW_DROP:  # second hand review (8 Nov): template errors, sample trains without a note
                dropped[sid] = {"tries": notes, "problems": ["review 2: " + ", ".join(REVIEW_DROP[sid])]}
            elif good:
                final[sid] = good[-1]
                keep.append(sid)
                words.append(len(good[-1].split()))
                tokens.append(len(c2.cn.tokenizer()(good[-1], add_special_tokens=False)["input_ids"]))
            else:
                dropped[sid] = {"tries": notes, "problems": probs[-1] if probs else ["no note returned"]}
        (GEM / f"batch_{b:03d}.jsonl").write_text(
            "".join(json.dumps({"id": s, "note": final[s]}, ensure_ascii=False) + "\n" for s in keep), encoding="utf-8")
    text = "".join(json.dumps({"id": i, "note": final[i]}, ensure_ascii=False) + "\n" for i in sorted(final))
    leaked = [v for v in validation if v in text]
    assert not leaked and not set(final) & validation, f"validation ids: {leaked[:5]}"
    (REPO / "src" / "lifemate" / "notes_v2.jsonl").write_text(text, encoding="utf-8")
    CHECKS.mkdir(parents=True, exist_ok=True)
    (CHECKS / "dropped.json").write_text(json.dumps(dropped, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(map(len, order))
    print(json.dumps({"samples": total, "first_pass_ok": first_ok, "final_notes": len(final), "dropped": len(dropped),
                      "dropped_pct": round(100 * len(dropped) / total, 2), "mean_words": round(sum(words) / len(words), 1),
                      "max_words": max(words), "mean_tokens": round(sum(tokens) / len(tokens), 1), "max_tokens": max(tokens),
                      "validation_ids_found": 0, "package": "src/lifemate/notes_v2.jsonl"}))


def cost() -> None:
    rows = read_jsonl(USAGE)
    by = {}
    for r in rows:
        key = (r["model"], r.get("thinking"))
        a = by.setdefault(key, [0, 0, 0, 0, 0.0])
        a[0] += 1; a[1] += r["prompt"]; a[2] += r["output"]; a[3] += r["thoughts"]; a[4] += r["usd"]
    for (m, th), (n, p, o, t, usd) in by.items():
        print(f"{m} thinking={th}: {n} calls, prompt {p}, output {o}, thoughts {t} tokens, {usd:.3f} USD")
    print(f"total {sum(r['usd'] for r in rows):.3f} USD in {len(rows)} calls")


def main() -> None:
    global KEY, SYSTEM, VALIDATION
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "fix", "assemble", "cost"])
    ap.add_argument("paths", nargs="*", help="input .txt files (write) or first-pass part files (fix); globs allowed")
    ap.add_argument("--name", default=None, help="output name for a single input (default gm_<input stem>)")
    ap.add_argument("--model", default="gemini-3.8-flash", choices=sorted(PRICES))
    ap.add_argument("--thinking", default=None, help="thinkingLevel: minimal, low, medium or high (default: the model's)")
    ap.add_argument("--per-call", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=6.0, help="no new call once usage.jsonl totals this much")
    ap.add_argument("--rounds", type=int, default=2, help="maximum fix rounds per part")
    ap.add_argument("--force", action="store_true", help="rewrite parts that exist")
    args = ap.parse_args()
    if args.cmd == "cost":
        return cost()
    if args.cmd == "assemble":
        return assemble()
    KEY, SYSTEM, VALIDATION = api_key(), system_text(), validation_ids()
    PARTS.mkdir(parents=True, exist_ok=True)
    paths = [Path(p).resolve() for pat in args.paths for p in sorted(glob.glob(pat))]
    if args.cmd == "fix":
        paths = [p for p in paths if not re.search(r"_fix\d+$", p.stem)]
    assert paths and (args.name is None or len(paths) == 1), "give input paths; --name only with one input"
    job = write_one if args.cmd == "write" else fix_one
    with ThreadPoolExecutor(args.jobs) as ex:
        for f in [ex.submit(job, args, p) for p in paths]:
            f.result()
    cost()


if __name__ == "__main__":
    main()
