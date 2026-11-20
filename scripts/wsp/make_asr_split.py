"""Fixed halves for the speech weight-space edit search (log.md, 11 Nov): train samples outside the 386 validation
samples, split by sample into discover (D1) and confirm (D2), balanced over the three speech folds (asr_train's
sample_folds, 3 folds, seed 2026). The 386 stay out: they are the test. Writes scripts/wsp/asr_split.json; if it
exists, checks that it is reproduced."""
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data  # noqa: E402
from lifemate.asr_train import sample_folds  # noqa: E402

items = data.load_split("train", data.find_root("train"))
splits = json.loads((REPO / "src" / "lifemate" / "splits.json").read_text())
val = set(splits["random"]) | set(splits["apps"])
folds = sample_folds(items, 3, 2026)
rng = random.Random(20261112)
out = {"discover": [], "confirm": [], "test": sorted(val)}
for k in range(3):
    ids = sorted(i["id"] for i in items if folds[i["id"]] == k and i["id"] not in val)
    rng.shuffle(ids)
    out["discover"] += sorted(ids[: len(ids) // 2])
    out["confirm"] += sorted(ids[len(ids) // 2:])
path = REPO / "scripts" / "wsp" / "asr_split.json"
if path.exists() and json.loads(path.read_text()) != out:
    raise SystemExit("asr_split.json differs from what this script makes")
path.write_text(json.dumps(out))
print({k: len(v) for k, v in out.items()})
