"""The 20-sample pilot input (notes_v2/inputs/pilot.txt): every kind, both branches of K1 and K2, all three generator batches."""
import random

from make_inputs_v2 import OUT, as_text, record
from notes_common import load_train
from notes_v2_common import analyze

items, catalog, validation = load_train()
pool = [i for i in items if i["id"] not in validation]
info = {i["id"]: analyze(i, catalog) for i in pool}
rnd = random.Random(11)
def pick(pred, n):
    c = [i for i in pool if pred(info[i["id"]], i)]
    return rnd.sample(c, n)
chosen = (pick(lambda f, i: f["primary"] == "K0" and not i["label"]["tool_calls"], 1)
          + pick(lambda f, i: f["primary"] == "K0" and f["ends_ask"], 1)
          + pick(lambda f, i: f["primary"] == "K0" and f["shape"] == "R", 1)
          + pick(lambda f, i: f["primary"] == "K0" and f["shape"] in ("W", "WW", "RW"), 2)
          + pick(lambda f, i: f["primary"] == "K1" and f["ends_ask"], 2)
          + pick(lambda f, i: f["primary"] == "K1" and not f["ends_ask"], 3)
          + pick(lambda f, i: f["primary"] == "K2" and f["any_write"], 3)
          + pick(lambda f, i: f["primary"] == "K2" and not f["any_write"], 3)
          + pick(lambda f, i: "K3" in f["kinds"], 2)
          + pick(lambda f, i: f["primary"] == "K4", 2))
assert len(chosen) == 20 and len({i["id"] for i in chosen}) == 20 and not {i["id"] for i in chosen} & validation
text = "\n".join(as_text(record(i, catalog)[0]) for i in chosen)
assert not [v for v in validation if v in text]
(OUT / "pilot.txt").write_text(text, encoding="utf-8")
print([ (i["id"], info[i["id"]]["primary"], info[i["id"]]["batch"]) for i in chosen])
