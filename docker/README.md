# LifeMate solution package

Voice requests in Vietnamese become tool calls on the organisers' `zaic_tools` environment:

```
audio → PhoWhisper-medium + five LoRA adapters (their mean = the main reading; each alone = five more readings of the latest request)
      → transcript with the readings view → prompt → Qwen3.5-4B text model (mean of two LoRA copies, merged) → agent loop with ToolEnv → result/submission.jsonl
```

Everything the container needs is inside the image: no network access, no external API.

## Run (inference)

```
docker load -i lifemate_final.tar.gz
docker run --rm --gpus all --network none -v /path/to/private_test:/data -v "$PWD/result:/result" lifemate:final
```

The default command is `bash /code/predict.sh`, so giving that command explicitly does the same.

- `/data` is the test split folder (same layout as `public_test`: `<split>.jsonl`, `audio/`, `<split>_tools_catalog.json`, `<split>_environment/`, `zaic_tools/`). A folder that contains the split folder also works.
- The result is written to `/result/submission.jsonl` (one line per sample, in the order of `<split>.jsonl`), plus `/result/timing.jsonl` with the seconds and number of tool calls of each sample. Both are also copied to `result/` under the working directory `/code`.
- The agent never opens `<split>_environment/`: all app data goes through the split's own `zaic_tools.ToolEnv`, exactly one per sample, and every call made on it is in the submission.
- Decoding is greedy for both models and all seeds are fixed (`--seed 2026`, `PYTHONHASHSEED=2026`), so a re-run gives the same file.
- The LLM decodes with prompt lookup (`--lookup 10`): up to 10 tokens copied from earlier in the conversation are checked in one forward pass, and every kept token is the model's own greedy token. On 193 validation samples it was 37 % faster (8.92 to 5.58 s mean) with the same score; 3 samples changed a word in free text (fp16 near-ties).
- Time: both models are loaded and warmed up once, then each sample is processed within its own budget: no new LLM step starts after 27 s from the start of the sample, and a step still decoding at 28.5 s stops there. Measured on all 634 `public_test` samples with these exact models and this code path (`solve.py`, Kaggle Tesla T4, 15 Nov 2026, kernel `lifemate-pkg-final`): **mean 9.13 s, p95 14.27 s, maximum 20.18 s per sample, none over 30 s (none over the 27 s budget), no hard stop, 0 errors**, model loading excluded. A T4 has 15 GB, so in that check the speech models sat on a second T4 (`--asr-device cuda:1`, a check-only flag); peak GPU memory of both models together was 17.14 GB reserved (16.33 GB allocated), well inside the A30's 24 GB. The A30 is faster than a T4.

## Models and size

| Part | Base (open source, released before 26 Oct 2026) | Fine-tuning | Parameters |
|---|---|---|---|
| ASR | `vinai/PhoWhisper-medium` | five LoRA r=32 adapters on all train audio (`models/asr_base` + `models/asr_view/0-4`; the averaged model is built from them at load time) | 763,857,920 base + 5 × 34,603,008 adapters = **936,872,960** |
| LLM | `Qwen/Qwen3.5-4B`, text model only (the vision tower is not shipped) | two LoRA r=32 copies (seeds 2026, 2027), each trained on train and then continued on all generated data (see Training 2b), their mean merged | **4,205,751,296** |
| Total | | | **5,142,624,256** (limit 6B; LLM limit 5B; context used at most 3,700 tokens in training, 16,384 allowed) |

## Training (reproducing the models)

All training ran on Kaggle (2 × T4) through `scripts/kg.py`, which pushes a script kernel that runs `python -m lifemate.jobs --plan <JSON>`: one chain of module commands per GPU. The plans are in `scripts/plans/`.

1. **ASR**: `lifemate.asr_train` fine-tunes PhoWhisper-medium with LoRA, three folds for out-of-fold transcripts of the train audio (so the LLM trains on realistic recognition errors) and one model on all train audio for the test split. `lifemate.merge_oof` joins the fold outputs. Five such recipes are shipped, in this order (`models/asr_view/0-4`): `exp7` (5 epochs), `medium10` (10 epochs), `B` (the `exp7` recipe plus synthetic speech, see Data), and two more seeds of `exp7` (2027, 2028). Their mean is the main reading; for the latest request the five single readings are also decoded (with the main reading as a draft, `asr.lookup_decode`) and shown in braces where they disagree (`view.py`, the readings view).
1b. **Reading chooser** (`lifemate/stack_view.py`, `models/stack_model.json`, 16 Nov 2026): where the five single readings of the latest request disagree with the averaged one, a conditional logistic regression (a softmax over the options of each disputed span; 15 features: how many and which readings gave the option, length, word frequency in the train transcripts, number words) picks the option put in the main reading; the other options stay in braces. Fitted on CPU on the out-of-fold train readings outside the validation split (4,015 disputed spans); on the 386 validation requests it lowered the word error rate from 8.64 % to 8.48 %. `predict.sh` adds `--stack-model` when the file is present.
2. **LLM**: `lifemate.train` fine-tunes Qwen3.5-4B with LoRA on the train conversations, with the out-of-fold transcripts in place of the true text, plus synthetic conversations. Final flags (`scripts/plans/view_full.json`, kernel `lifemate-view-full-r2`, one copy per seed 2026 and 2027): `--transcripts view:avg5_train.json,medium_train.json,medium10_train.json,b_train.json,s2027_train.json,s2028_train.json --noisy-only --tool-drop 0.5 --label-history --synth package --denoise --epochs 2.0 --max-hours 11.2 --mask-names 0.3 --notes-aux package-v2`. The two copies are averaged (mean of the LoRA deltas) at export.
2b. **LLM, continuation on all generated data** ("p3all", the final model): each copy above continues training for one pass over the 7,805 generated conversations and exercises described in Data (sets T2a-T10), mixed with replayed train conversations (60 % of the stream) and the side task (15 % of the replay), 19,513 items = 4,879 steps per copy. Plans `scripts/plans/p3all_a2026_s1.json` / `p3all_a2027_s1.json` (session 1) and `_s2.json` (sessions 2 and 3, `--resume` from the previous session's `state.pt`), run with `python -m lifemate.ddp train` on both T4s: the flags of step 2 plus `--lr 1e-4 --init-adapter v2026/adapter` (or `v2027`) `--extra t2a_full.jsonl,...,t9_full.jsonl,t10_k2.jsonl --replay-share 0.6 --replay-side 0.15 --seed 2026` (or 2027). Kernels `lifemate-p3all-a`, `-a2-r3`, `-a3` and `lifemate-p3all-b-r2`, `-b2-r2`, `-b3`; both copies finished all 4,879 steps (11 h + 11 h + 3 h each); 8 and 11 items longer than `--max-seq-len` were skipped by design.
3. **Export**: `python -m lifemate.export_models --out models --llm-adapter a2026/adapter,a2027/adapter --asr-model vinai/PhoWhisper-medium --asr-view-adapters asr_medium/full/best+asr_m10/full/best+b/full/best+s2027/full/best+s2028/full/best` (`scripts/plans/pkg_final.json`, kernel `lifemate-pkg-final`): the LLM in half precision with the mean delta merged (`models/llm`), the speech base and its five adapters (`models/asr_base`, `models/asr_view/`), and `models/sha256.json`, `models/parameters.json`.
4. **Image**: `docker build -f docker/Dockerfile -t lifemate:final .` from the repository root, with the exported models in `./models`.

### Data

- The organisers' `train` split. `public_test` and `private_test` were never labelled or used to train anything; no test request text or audio was used to train or tune. The only test-side inputs used in development: the `public_test` tools catalog (tool schemas, to write synthetic conversations for apps that train lacks) and the public leaderboard scores of finished models, which we uploaded by hand and used to choose between them.
- Synthetic conversations, generated by code in this package from the tool catalogs and the organisers' simulator, never from test requests:
  - `src/lifemate/synth_samples.jsonl` (`scripts/make_synth.py`, in the final model): apps that appear in `public_test` but not in `train`. The request templates and value lists are in the script; the code was written with an AI coding assistant (Claude Code).
  - `src/lifemate/synth_held.jsonl` (`scripts/make_synth_held.py`): used only in a validation experiment (E2).
- Side task in the final model (`--notes-aux package-v2`, `src/lifemate/notes_v2.jsonl`): 1,148 short notes on train samples (which tools a request needs and why), **written by Google Gemini 3.8 Flash** through its API during development (`scripts/notes/gemini_notes.py`), checked by code; notes that failed the checks were sent back to Gemini or dropped, never edited by hand or by another model. They are training targets of an auxiliary turn only and are not used at inference.
- Synthetic speech for the ASR adapter `B`: train request texts with names, places and numbers swapped by code (`scripts/make_synth_asr.py`, `data/synth_asr/texts.json`), spoken by the open-source VieNeu-TTS model; used only to train that adapter.
- **Generated sets in the final model** (step 2b; generators in `scripts/p3/`, each filtered by the checks in `scripts/p3/data_gate.py`; built from the train split and the tool catalogs, never from test requests):
  - written by code (generators written with an AI coding assistant, Claude Code): T2a (train conversations with other out-of-fold speech readings of the same audio), T2b (simulated mishearings from a confusion table measured on train), T3 (tool-choice exercises), T4 (requests with a required value removed, answer = ask for it), T5 (the needed app not installed, answer = no call), T6 (more conversations for apps that appear in the public tools catalog but not in train, extending `make_synth.py`), T8 (argument values swapped in train conversations);
  - written by **Google Gemini** (through the user's own account during development; checked by code, never edited by hand or by another model): T7 (rewordings of train requests in northern, central and southern speech), T9 (new requests), T10 (1,149 conversations for the eleven apps of the public catalog that train lacks, `scripts/p3/make_t10.py`).
  On a held-out fold these sets lowered the validation score (log of 14 Nov 2026); trained on all of them, the final model scored 73.98 on the public leaderboard against 69.72 for the same model before step 2b, so it was kept.
- Tried and not in the final model: three more speech adapters trained with 6,000 clips of the FPT Open Speech Dataset (FOSD, CC BY 4.0, FPT 2018, Hugging Face mirror `doof-ferb/fpt_fosd`; `src/lifemate/ext_speech.py`, `scripts/plans/asr_ext_*.json`). They heard train audio worse on a held-out fold (word error rate 9.06 % to 10.35 % on the latest requests) and were dropped.
- `src/lifemate/splits.json`: the fixed validation split (202 random samples and the 184 samples of six apps held out of training).

## Files (under `/code` in the image)

- `predict.sh`: the default command; runs `lifemate/solve.py` with the final flags.
- `lifemate/solve.py`: audio to `submission.jsonl` for a whole split.
- `lifemate/infer.py`, `lifemate/prompts.py`: the agent loop and the prompt, shared by training and inference.
- `lifemate/asr.py`: speech recognition.
- `scripts/`: Kaggle orchestration (`kg.py`, `plans/`), synthetic data generators, validation comparison (`compare_runs.py`).
- `data/readings/`: the six transcript sets the final LLM was trained with (out-of-fold readings of the train audio: the five-adapter mean `avg5` and the single adapters) and the same readings of `public_test` that its leaderboard file used; `data/synth_asr/`: the synthetic speech texts.
- `models/llm`: the merged LLM; `models/asr_base`, `models/asr_view/0-4`: the speech base and its five adapters.
- `tests/`: unit tests (`python -m pytest tests`, with `LIFEMATE_DATA` pointing at the folder that holds `train`).
