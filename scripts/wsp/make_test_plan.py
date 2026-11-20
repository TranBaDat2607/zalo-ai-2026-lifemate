"""Plan for the test stage (log.md, weight-space programming, rule 4): a confirmed edit program applied to a view-recipe
fold model, scored on that fold's samples with exactly the flags of the model's own evaluation.

    python scripts/wsp/make_test_plan.py '{"all": 1.15}' --fold 0 --adapters auto:v1,auto:v2 > scripts/plans/wsp_test_cv0.json
    python scripts/wsp/make_test_plan.py '{"all": 1.15}' --fold 1 --public   # all-data public run with the program
    python scripts/wsp/make_test_plan.py none --ask-bias -1.0 --fold 0       # the ask push, weights unedited
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lifemate.wsp import parse_program  # noqa: E402

VIEW = ("view:auto:avg5_{s}.json,auto:medium_{s}.json,auto:medium10_{s}.json,auto:b_{s}.json,auto:s2027_{s}.json,"
        "auto:s2028_{s}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("program")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--adapters", default="auto:v1,auto:v2")
    ap.add_argument("--public", action="store_true", help="public_test split, all ids (the all-data model)")
    ap.add_argument("--ask-bias", default=None, help="also pass infer --ask-bias (the decision-level edit, log.md 11 Nov)")
    ap.add_argument("--with-base", action="store_true", help="also rerun the unedited model in the same session")
    ap.add_argument("--exclude-dpo1", action="store_true",
                    help="score only the fold's samples outside folds_dpo fold 1 (the discovery data), as an inline id list")
    args = ap.parse_args()
    prog = json.dumps(parse_program(args.program))
    split = "public_test" if args.public else "train"
    arms = [("edit_eval", prog)] + ([("base_eval", "none")] if args.with_base else [])
    plan = {}
    for g in ("0", "1"):
        chain = []
        for name, p in arms:
            cmd = ["infer", "--out", "{out}/" + name, "--split", split, "--adapter", args.adapters, "--edit", p,
                   "--transcripts", VIEW.format(s="public_test" if args.public else "train"), "--budget-s", "60",
                   "--denoise", "--lookup", "10", "--shard", g, "--shards", "2"]
            if not args.public:
                ids = "splits" if args.fold == 0 else f"cvscore:{args.fold}"
                if args.exclude_dpo1:
                    from lifemate import data
                    ids = json.dumps(sorted(data.id_set(ids) - data.id_set("fold:1")))
                cmd[cmd.index("--budget-s"):cmd.index("--budget-s")] = ["--ids-file", ids]
            if args.ask_bias is not None and name == "edit_eval":
                cmd += ["--ask-bias", args.ask_bias]
            chain.append(cmd)
        plan[g] = chain
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
