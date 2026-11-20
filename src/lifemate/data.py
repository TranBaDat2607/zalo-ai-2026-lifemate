"""Locate the data package locally or on Kaggle, load splits and import each split's zaic_tools."""
import importlib
import json
import os
import sys
from pathlib import Path

LOCAL_ROOT = Path(__file__).resolve().parents[2] / "train+public_test"
KAGGLE_INPUT = Path("/kaggle/input")
_SKIP_DIRS = {"audio", "zaic_tools"}
# id_set("cv:K") = what a fold-K model is not trained on; "cvscore:K" = what it is scored on, in two parts.
_CV_PARTS = {"cv": ("holdout",), "cvscore": ("held", "random"), "cvheld": ("held",), "cvrandom": ("random",)}


def walk_input(filename: str, base: Path = KAGGLE_INPUT):
    """Yield folders under ``base`` holding ``filename``; the mount layout of Kaggle datasets varies."""
    for d, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x not in _SKIP_DIRS and not x.endswith("_environment")]
        if filename in files:
            yield Path(d)


def find_root(split: str) -> Path:
    """Folder holding ``<split>/``, which is the ``root`` that ToolEnv expects."""
    if os.environ.get("LIFEMATE_DATA"):
        return Path(os.environ["LIFEMATE_DATA"])
    if (LOCAL_ROOT / split / f"{split}.jsonl").is_file():
        return LOCAL_ROOT
    for d in walk_input(f"{split}.jsonl"):
        return d.parent
    raise FileNotFoundError(f"no {split}.jsonl locally or under {KAGGLE_INPUT}; set LIFEMATE_DATA")


def load_split(split: str, root: Path) -> list:
    with open(root / split / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def id_set(spec: str) -> set:
    """Sample ids from a JSON file, an inline JSON list, or 'splits' for the fixed validation split (random part plus held-out apps)."""
    if spec.startswith("["):
        return set(json.loads(spec))
    if spec == "splits":
        splits = json.loads((Path(__file__).parent / "splits.json").read_text(encoding="utf-8"))
        return set(splits["random"]) | set(splits["apps"])
    if spec.startswith("fold:"):  # the 3 folds outside the validation set (scripts/make_dpo_folds.py)
        folds = json.loads((Path(__file__).parent / "folds_dpo.json").read_text(encoding="utf-8"))
        return set(folds[spec.split(":", 1)[1]])
    kind, _, fold = spec.partition(":")
    if kind in _CV_PARTS and fold:  # four-fold cross-validation, fold 0 = 'splits' (scripts/make_cv_folds.py)
        folds = json.loads((Path(__file__).parent / "folds_cv.json").read_text(encoding="utf-8"))
        return set().union(*(folds[part][fold] for part in _CV_PARTS[kind]))
    return set(json.loads(Path(spec).read_text(encoding="utf-8")))


def load_catalog(split: str, root: Path) -> dict:
    """Tool specs by name, for every app of the split."""
    with open(root / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        return {t["name"]: t for app in json.load(f)["apps"] for t in app["tools"]}


def import_tool_env(split: str, root: Path):
    """ToolEnv of this split. Each split ships its own zaic_tools, so an already imported one is dropped."""
    for name in [m for m in sys.modules if m == "zaic_tools" or m.startswith("zaic_tools.")]:
        del sys.modules[name]
    sys.path.insert(0, str(root / split))
    try:
        return importlib.import_module("zaic_tools").ToolEnv
    finally:
        sys.path.pop(0)
