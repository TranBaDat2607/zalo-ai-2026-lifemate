# LifeMate: a Vietnamese voice agent that turns requests into tool calls

Our solution for the **LifeMate** track of the [Zalo AI Challenge 2026](https://challenge.zalo.ai). The input is a
spoken Vietnamese request, plus the user's context, the earlier turns of the conversation and the apps installed on the
device. The output is an ordered list of tool calls on the organisers' simulated `zaic_tools` environment. When a
required value is missing, the output is a call to `sys.ask_missing_info`; when there is nothing to do, it is an empty
list. A request scores 0 unless the final app state matches the reference; otherwise it scores by how closely the calls
match the reference calls.

The final solution is one offline Docker image:

- **5.14 B parameters** in total (the limit is 6 B);
- about **9–11 s per request** on a T4 GPU (the limit is 30 s);
- no network access and no external API.

| | Score |
|---|---|
| Public leaderboard (634 requests) | **73.98** |
| Private test (625 requests) | **59.02** |

The private score is much lower than the public one. The likely reason, and what we learned from it, is in
[Public vs private](#public-vs-private).

## How it works

```
audio ─► PhoWhisper-medium + five LoRA adapters ─► transcript + readings view
      ─► prompt ─► Qwen3.5-4B + LoRA ─► restating step ─► agent loop on ToolEnv ─► submission.jsonl
```

- **Speech.**
  - `vinai/PhoWhisper-medium`, fine-tuned with LoRA (`asr.py`, `asr_train.py`).
  - The shipped speech model is the mean of five adapters.
  - For the latest request, the five adapters' separate readings are also decoded and shown in braces where they
    disagree, e.g. `{thứ|tối} mai` (`view.py`). The language model then picks between plausible hearings.
- **Agent.**
  - `Qwen/Qwen3.5-4B` (text part only), fine-tuned with LoRA on the train conversations.
  - Training uses out-of-fold speech transcripts: each conversation is transcribed by a speech model that never trained
    on it. The agent therefore learns from realistic recognition errors, not from clean text.
  - Two trained copies are averaged into one model (a "soup").
- **Restating step.** The first assistant turn restates what the user really said, then calls tools. Without this step,
  the same model scored 62.00 on public instead of 69.72.
- **Extra training data.**
  - **Generated conversations:**
    - conversations for apps that train lacks;
    - simulated mishearings;
    - exercises on choosing the right tool and on asking for a missing value;
    - rewordings of requests in northern, central and southern speech.
  - Some were written by code and some by Google Gemini (details in [`docker/README.md`](docker/README.md#data)).
  - The final model continues training for one pass over 7,805 of these conversations.
- **Inference** (`solve.py`, `infer.py`):
  - greedy decoding with prompt-lookup drafts (faster, same output);
  - fixed seeds;
  - one `ToolEnv` per request;
  - a time budget per request.
- **One prompt format.** Training and inference render prompts through the same code (`prompts.py`).

## Results

| Model | Validation (fold 0) | Public | Private |
|---|---|---|---|
| Without the restating step | — | 62.00 | — |
| Two-copy soup, five-adapter speech, readings view (`0019`) | 71.57 (fold-0 twin) | 69.72 | — |
| `0019` + 7,805 generated conversations (**final**) | — | **73.98** | **59.02** |

- **Validation split.** It is fixed in `src/lifemate/splits.json`: 202 random train requests, plus 184 requests from six
  apps held out of training.
  - Models trained without that split are scored on it.
  - Models trained on all data could only be judged on the public leaderboard.
- **Timing.** These are times of the final package on Kaggle T4s; there were no errors in either run.

  | Run | Mean | Slowest | Requests over 30 s |
  |---|---|---|---|
  | Public | 9.1 s | 20.2 s | 0 |
  | Private | 11.4 s | 22.0 s | 0 |

### What helped and what did not

These changes were tested and kept:

| Change | Effect |
|---|---|
| Restating step | public 62.00 → 69.72 |
| Averaging two trained copies | +2.4 on validation |
| Five-adapter speech with the readings view | kept |
| Side task of short tool notes during training | kept |
| Generated data for the public catalog's apps | public +4.3 (see below) |

These changes were tested and dropped:

| Change | Effect |
|---|---|
| Step-level preference training (DPO) | −12 to −26 on validation |
| Number and time drills | −1.3 on three folds |
| Rescaling the LoRA weights | no gain |
| Learned chooser over the five speech readings | public −1.34 |

## Public vs private

The private set differs from public in ways that public never tested.

| | Public | Private |
|---|---|---|
| Apps in the tools catalog | 36 | 65 |
| Requests touching an app with no training data at all | ~0 % | ~51 % |
| Requests with 8 or more earlier user turns | 7 % | 37 % |
| Share of the hardest batch (C) | 40 % | 47 % |

Public's speech audio and private's look the same: clip lengths, loudness and the 16 kHz sample rate all match. So the
drop is on the agent's side, not the speech side.

- **Apps never seen in training.**
  - Public added 11 apps that train lacks. Our generated data covered all 11, and public rose by 4.3 points.
  - Private added 8 more, such as hotels, clinic and restaurant booking, vehicle and cloud files. No training data of
    any kind covered them.
  - Suppose the requests on known apps scored about 71, as our validation fit predicts. Then the requests that touch
    the 8 new apps would have averaged in the high 40s.
  - This is an estimate: the private answers are not released, so it cannot be checked request by request.
- **Our prediction.** Before the private score was out, we predicted about 69, with a range of 64–74. We based the cost
  of unseen apps on the six apps held out of validation, which cost 3–8 points per request. That stand-in was too
  optimistic. The held-out apps came from the same generator as train and were simpler, while several new private apps
  need a search → choose → book flow.
- **Lesson.** Generated data for the apps you know about is not the same as generalising to apps you have not seen. To
  measure real generalisation, hold out apps with no generated data for them either, and include multi-step apps among
  them.

## Data

The organisers' data package is **not included and must not be redistributed**; it was given to registered teams
only. To run anything, put the package next to the repository and point `LIFEMATE_DATA` at the folder that holds
`train/` and `public_test/`:

```
<LIFEMATE_DATA>/train/train.jsonl, train_tools_catalog.json, train_environment/, audio/, zaic_tools/
<LIFEMATE_DATA>/public_test/...      (same layout; private_test/ likewise)
```

## Run the final solution

Build the image from the repository root, with the exported models in `./models`. The models are exported with
`python -m lifemate.export_models`; see [`docker/README.md`](docker/README.md). Then run it on a test split:

```bash
docker build -f docker/Dockerfile -t lifemate:final .
docker run --rm --gpus all --network none \
  -v /path/to/private_test:/data -v "$PWD/result:/result" lifemate:final
```

The output is `result/submission.jsonl`, in the order of the split file, plus `timing.jsonl` with per-request times.

## Reproduce training

All training ran on Kaggle kernels (2 × T4); the development machine has no GPU.

- `scripts/kg.py` uploads the code and data as private Kaggle datasets.
- It then pushes a kernel that runs `python -m lifemate.jobs --plan <JSON>`, which runs one chain of `lifemate.*`
  modules per GPU.
- The plans are in `scripts/plans/`.
- The recipe for every shipped model is in
  [`docker/README.md`](docker/README.md#training-reproducing-the-models).

```bash
conda env create -f environment.yml && conda activate lifemate      # CPU-only development environment
export LIFEMATE_DATA=/path/to/data                                  # the organisers' data package
python -m pytest -q tests                                           # unit tests (about 1 minute, CPU)

python scripts/kg.py code -m "msg"                                  # upload src/lifemate as a private Kaggle dataset
python scripts/kg.py push --hours 12 NAME jobs -- --plan "$(tr -d '\n' < scripts/plans/PLAN.json)"
python scripts/kg.py status|wait|collect NAME
```

Options of `kg.py push` (`--hours`, `--dataset`, `--from-kernel`) go **before** the kernel name. Anything after the
module name is passed to the module.

## Repository layout

| Path | Contents |
|---|---|
| `src/lifemate/` | the package |
| `scripts/` | Kaggle orchestration (`kg.py`, `plans/`), generators for the synthetic data (`make_synth*.py`, `p3/`), CPU analyses of saved runs |
| `docker/` | `Dockerfile`, `predict.sh` and the package README |
| `tests/` | unit tests |
| `solver/`, `predict.sh` | a different, rule-based track (DataMate); not part of LifeMate |

Main modules in `src/lifemate/`:

| Area | Modules |
|---|---|
| Speech | `asr*.py`, `view.py` |
| Prompt and agent loop | `prompts.py`, `infer.py`, `solve.py` |
| Training | `train.py`, `stream.py`, `ddp.py` |
| Kaggle job runner | `jobs.py` |
| Local copy of the scorer | `scorer.py` |
| Model export | `export_models.py` |

## Rules followed

- Only open-source base models released before 26 Oct 2026 (PhoWhisper-medium, Qwen3.5-4B).
- Training used the organisers' train split and synthetic data.
- `public_test` and `private_test` were never labelled, and never used for training or tuning. The only test-side
  inputs were the public tools catalog (for generated conversations about its apps) and the public leaderboard scores.
- No network access and no external model or API at inference.
- Use of AI tools:
  - Google Gemini wrote some of the training data: the notes and some of the generated conversations.
  - An AI coding assistant (Claude Code) helped write the code and the data generators.
  - No model wrote reasoning text that was used as training targets, apart from the Gemini notes described in
    [`docker/README.md`](docker/README.md#data).

## Acknowledgements

- [VinAI PhoWhisper](https://github.com/VinAIResearch/PhoWhisper)
- [Qwen](https://github.com/QwenLM)
- Hugging Face `transformers` and `peft`
- Kaggle, for the free GPUs
- The Zalo AI Challenge organisers
