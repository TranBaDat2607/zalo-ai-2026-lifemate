"""Plan 3 stream training (lifemate.stream, train --extra, lifemate.ddp): order, gradient sum, resume and two processes.

The end-to-end tests train a tiny random Llama with the cached Qwen3.5 tokenizer on CPU: an uninterrupted run, a run
stopped after its first step and resumed, and the same run on two processes must end with the same adapter.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lifemate import stream  # noqa: E402

DATA = Path(os.environ.get("LIFEMATE_DATA", REPO / "train+public_test" / "train+public_test"))


def test_order_has_every_extra_once_and_the_replay_share():
    order = stream.build_order(1000, 300, 50, 0.2, 0.15, seed=3)
    extra = [i for k, i in order if k == "x"]
    assert sorted(extra) == list(range(1000))
    replay = [x for x in order if x[0] != "x"]
    assert len(replay) == 250  # 20 % of 1250
    assert sum(k == "s" for k, _ in replay) == round(250 * 0.15)
    mains = [i for k, i in order if k == "m"]
    assert max(mains.count(i) for i in set(mains)) - min(mains.count(i) for i in range(300)) <= 1  # even replay
    assert order == stream.build_order(1000, 300, 50, 0.2, 0.15, seed=3)
    assert order != stream.build_order(1000, 300, 50, 0.2, 0.15, seed=4)
    assert stream.build_order(10, 0, 0, 0.2, 0.15, seed=1) == [x for x in stream.build_order(10, 0, 0, 0.2, 0.15, seed=1)]
    assert all(k == "x" for k, _ in stream.build_order(10, 0, 0, 0.2, 0.15, seed=1))


def test_positions_split_a_step_between_processes():
    parts = [stream.my_positions(40, 4, r, 2) for r in range(2)]
    assert parts == [[40, 42], [41, 43]]
    assert stream.my_positions(40, 3, 1, 2) == [41]
    assert stream.my_positions(8, 4, 0, 1) == [8, 9, 10, 11]


def _worker(rank, world, port, batch, target, out):
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": str(rank), "WORLD_SIZE": str(world)})
    stream.init_distributed()
    torch.manual_seed(0)
    model = torch.nn.Linear(5, 3)
    params = list(model.parameters())
    for p in stream.my_positions(0, len(batch), rank, world):
        loss = torch.nn.functional.mse_loss(model(batch[p]), target[p]) / len(batch)
        loss.backward()
    stream.allreduce_grads(params, world)
    if rank == 0:
        torch.save([p.grad for p in params], out)
    import torch.distributed as dist
    dist.destroy_process_group()


def test_gradient_sum_over_two_processes_equals_one_process(tmp_path):
    from lifemate.ddp import free_port
    torch.manual_seed(1)
    batch, target = torch.randn(4, 5), torch.randn(4, 3)
    torch.manual_seed(0)
    model = torch.nn.Linear(5, 3)
    for i in range(4):
        (torch.nn.functional.mse_loss(model(batch[i]), target[i]) / 4).backward()
    one = [p.grad for p in model.parameters()]
    out = tmp_path / "g.pt"
    torch.multiprocessing.spawn(_worker, args=(2, free_port(), batch, target, out), nprocs=2)
    two = torch.load(out)
    for a, b in zip(one, two):
        assert torch.allclose(a, b, atol=1e-6)


def test_launcher_restarts_all_processes_and_reports_failures(tmp_path):
    pkg = tmp_path / "ddpprobe"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "mark = Path(sys.argv[1]) / f\"r{os.environ['RANK']}\"\n"
        "if sys.argv[2] == 'fail' and os.environ['RANK'] == '1': sys.exit(3)\n"
        "if not mark.exists():\n    mark.write_text(os.environ['WORLD_SIZE']); sys.exit(75)\n")
    env = {**os.environ, "PYTHONPATH": f"{tmp_path}:{REPO / 'src'}", "CUDA_VISIBLE_DEVICES": ""}
    ok = subprocess.run([sys.executable, "-m", "lifemate.ddp", "--nproc", "2", "ddpprobe.mod", str(tmp_path), "ok"], env=env)
    assert ok.returncode == 0  # first launch: both exit 75; second: both 0
    assert (tmp_path / "r0").read_text() == "2" and (tmp_path / "r1").exists()
    bad = subprocess.run([sys.executable, "-m", "lifemate.ddp", "--nproc", "2", "ddpprobe.mod", str(tmp_path), "fail"], env=env)
    assert bad.returncode == 3


# --- end to end on a tiny model ---------------------------------------------------------------------------------------

def _tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    if not (DATA / "train" / "train.jsonl").exists():
        pytest.skip("competition data not here")
    tok = _tokenizer()
    if tok is None:
        pytest.skip("Qwen3.5 tokenizer not cached")
    from transformers import LlamaConfig, LlamaForCausalLM
    from lifemate import data, prompts
    root = tmp_path_factory.mktemp("stream")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=8192, tie_word_embeddings=True)
    LlamaForCausalLM(cfg).save_pretrained(root / "llm")
    tok.save_pretrained(root / "llm")
    rows = data.load_split("train", DATA)
    keep = rows[:12]
    (root / "hold.json").write_text(json.dumps([r["id"] for r in rows[12:]]))
    extras = [{"kind": "conv", "source": r["id"], "item": r, "texts": prompts.gold_transcripts(r)} for r in keep[:6]]
    packs = [json.loads(line) for line in (REPO / "src" / "lifemate" / "drills_cv1.jsonl").read_text().splitlines()[:2]]
    extras += [{"kind": "drill", "pack": p} for p in packs]
    with open(root / "extra.jsonl", "w", encoding="utf-8") as f:
        for e in extras:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return root


def _args(root, out, *more):
    return ["--llm", str(root / "llm"), "--out", str(out), "--holdout-file", str(root / "hold.json"),
            "--extra", str(root / "extra.jsonl"), "--grad-accum", "2", "--lora-r", "4", "--lora-dropout", "0",
            "--denoise", "--seed", "1", "--save-every", "2", "--replay-share", "0.3", "--lr", "1e-3", *more]


def _run(root, out, *more, nproc=1):
    env = {**os.environ, "LIFEMATE_DATA": str(DATA), "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO / "src"),
           "HF_HUB_OFFLINE": "1"}
    cmd = ([sys.executable, "-m", "lifemate.ddp", "--nproc", str(nproc), "train"] if nproc > 1
           else [sys.executable, "-m", "lifemate.train"]) + _args(root, out, *more)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    return json.loads((out / "metrics.json").read_text())


def _lora(out):
    from safetensors.torch import load_file
    return load_file(str(out / "adapter" / "adapter_model.safetensors"))


def _close(a, b):
    assert a.keys() == b.keys()
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    assert worst < 1e-4, worst
    assert any(float(v.abs().sum()) > 0 for k, v in a.items() if "lora_B" in k)  # it trained


def test_stream_run_resume_and_two_processes_agree(setup):
    root = setup
    full = _run(root, root / "full")
    assert full["finished"] and full["items"] == full["of_items"] and full["steps"] == full["total_steps"]
    assert full["extra"] == 8 and full["of_items"] > 8

    part = _run(root, root / "part", "--max-hours", "0")  # stops after its first step, state saved
    assert not part["finished"] and part["steps"] == 1
    at2 = _run(root, root / "at2", "--stop-step", "2")  # stops after step 2, state saved
    assert not at2["finished"] and at2["steps"] == 2 and at2["steps"] < full["steps"]
    resumed = _run(root, root / "resumed", "--resume", str(root / "part"))
    assert resumed["finished"] and resumed["steps"] == full["steps"]
    assert [s["started_step"] for s in resumed["sessions"]] == [0, 1]
    _close(_lora(root / "full"), _lora(root / "resumed"))

    two = _run(root, root / "two", nproc=2)
    assert two["finished"] and two["processes"] == 2 and two["steps"] == full["steps"]
    _close(_lora(root / "full"), _lora(root / "two"))


def test_init_adapter_continues_the_given_weights(setup):
    root = setup
    if not (root / "full" / "adapter").exists():
        _run(root, root / "full")
    cont = _run(root, root / "cont", "--init-adapter", str(root / "full" / "adapter"), "--lr", "0")
    _close(_lora(root / "full"), _lora(root / "cont"))  # lr 0: the loaded weights come back unchanged
    assert cont["finished"]


def test_resume_refuses_another_pass(setup):
    root = setup
    if not (root / "part" / "state.pt").exists():
        _run(root, root / "part", "--max-hours", "0")
    env = {**os.environ, "LIFEMATE_DATA": str(DATA), "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO / "src"),
           "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, "-m", "lifemate.train"] + _args(root, root / "other", "--resume", str(root / "part"),
                                                                        "--replay-share", "0.5"),
                       env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode != 0 and "another pass" in (p.stdout + p.stderr)
