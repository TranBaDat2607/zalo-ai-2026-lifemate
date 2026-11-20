"""Encode every final plan-3 row (outputs/p3/final/*_<pool>.jsonl) exactly as train.py's stream does; report errors,
rows over 3,700 tokens and Qwen token totals.  Usage: python scripts/p3/encode_check.py full|fold0"""
import json, random, sys, collections
sys.path.insert(0, "src")
from lifemate import data, prompts
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
tok = lambda t: tk(t, add_special_tokens=False)["input_ids"]
cat = {**data.load_catalog("public_test", data.find_root("public_test")), **data.load_catalog("train", data.find_root("train"))}
pool = sys.argv[1]
HOK = prompts.final_call_policy(data.load_split("train", data.find_root("train")))
rep = {}
from pathlib import Path
for f in sorted(Path("outputs/p3/final").glob(f"*_{pool}.jsonl")):
    n = err = over = 0; toks = 0; lab = 0; errs = collections.Counter()
    for k, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
        r = json.loads(line); n += 1
        try:
            if r["kind"] == "conv":
                ids, labels = prompts.encode_conversation(r["item"], r["texts"], cat, tok, True, 3700, 0.5, random.Random(k), HOK, None, None)
            else:
                ids, labels = prompts.encode_tool_pack(r["pack"], cat, tok)
        except Exception as e:
            err += 1; errs[type(e).__name__ + ": " + str(e)[:80]] += 1; continue
        if len(ids) > 3700: over += 1
        toks += len(ids); lab += sum(1 for x in labels if x != -100)
    rep[f.stem] = {"rows": n, "errors": err, "over_3700": over, "tokens_M": round(toks / 1e6, 2), "trained_tokens_M": round(lab / 1e6, 2), "err_examples": dict(errs.most_common(3))}
    print(f.stem, rep[f.stem], flush=True)
print("TOTAL", round(sum(v["tokens_M"] for v in rep.values()), 2), "M tokens,", sum(v["rows"] for v in rep.values()), "rows")
