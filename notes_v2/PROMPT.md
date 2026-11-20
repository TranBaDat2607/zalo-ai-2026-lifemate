# Task: worked notes v2 — rewrite the notes so they state rules, not outcomes

You are working in the repository root (Zalo AI Challenge 2026, LifeMate: spoken Vietnamese
request → tool calls, scored on the final app state and call match). The deadline for the final Docker solution is
16 Nov 2026, 23:59 GMT+7. Use GMT+7 for all times. Talk to the user in plain language; put technical detail in
`log.md`.

Your job is **only to produce a new, leak-free set of worked notes and the tools that check them**. You do not train,
you do not push anything to Kaggle, you do not upload, and you do not commit. When the notes are done and checked, you
stop and report.

Another Claude session works in this repo at the same time (Kaggle runs, public files). Do not touch `outputs/`,
`scripts/plans/` files other than the one you create, or Kaggle. Keep your running log in `notes_v2/LOG.md`; at the
end, add your summary section to `log.md` by appending only (never rewrite or reorder `log.md`; the other session
appends to it too).

## 1. Background (read these first)

- `log.md`, these sections: "Worked notes: design, written before any training", "Worked-notes fold run: result,
  DROP", "Why the notes helped batch A and hurt batch C", "The user's idea: fix the leaky B/C notes…".
- `notes/GUIDE.md` (the v1 writer guide), `scripts/notes/*.py` (v1 tools: `notes_common.py`, `make_inputs.py`,
  `check_notes.py`, `assemble.py`, `read_leak_scan.py`, `build_package_notes.py`, `why_notes.py`),
  `tests/test_notes.py`, and `src/lifemate/train.py` (`--notes`).

What a note is: one short Vietnamese line that the model writes inside its first thinking block of the final turn,
right after restating the request, before any tool runs. v1 notes exist for the 1,159 train samples outside validation
(`src/lifemate/notes.jsonl`, `notes/batch_*.jsonl`).

What happened with v1 (validation, 386 samples, notes pair vs the same recipe without notes): 66.01 vs 66.71.
- Where the key acts and nothing has to be looked up first: **+5.3** (14 better, 6 worse). Time arithmetic and
  multi-step plans help, also plans whose reads only supply a value (a bill amount, which account has enough):
  batch A, which is mostly such requests, gained +8.2.
- Where the key looks something up and then asks: **−16.0** (1 better, 9 worse); in batch C −22.4 (0/7).
- Ask/act: the notes model asks correctly on 19 of 39 ask samples (no-notes model: 27) and acts where the key asks 17
  times (no-notes: 10).

Why — four causes, counted on the 40 B/C samples the notes made worse:
1. **Outcome leaks (7).** The writer saw the answer key, so some notes state what a read will show.
   `"không xác định được thì sys.ask_missing_info"` is in 72 notes (A 51, B 6, C 15) and the key asks in all 72;
   `"chỉ đọc"` is in 30 notes, 29 of them read-only answers, 12 of them conditional requests whose condition turned out
   false. The model learned to guess the outcome in its note and then follow the guess.
2. **Deciding before the lookup (9).** Ordinary notes such as "subscriptions.list tìm gói, rồi subscriptions.cancel"
   commit to acting before the read. When the read is ambiguous, the key asks, but the model acts. 801 training notes
   are of this lookup-first kind (A 299, B 157, C 345).
3. **Invented details (9).** Notes mention values or options the key never uses (the current time on a workout log,
   `delay 60 phút`, a `merchant` filter, "xăng 92" for a search for "xăng", a full address for `near`, `action
   archive` copied from the tool name `email.archive` when the key's action is `delete`). The model copies them into
   arguments.
4. **The model's own reasoning slips (13).** Not fixable by note wording; out of scope.

## 2. Hard rules (non-negotiable)

- **Validation is sacred.** The 386 validation ids are in `src/lifemate/splits.json` (`random` + `apps`). Never read
  their data, labels, v1 notes or model outputs; never put a validation id in any input, note, guide or example; every
  script that writes inputs or notes must assert this. `log.md` mentions validation ids as examples of failures; use
  them only to understand the failure kinds, never as guide examples. Take guide examples from non-validation train
  samples or make them up.
- Never open anything under `public_test/`'s `public_test_environment/` or any `state.json` of public test. Train
  labels (`label.tool_calls`, `tool_results`) may be used to build inputs and checks, as in v1.
- The note writers are Claude subagents only. No web, no other APIs or models.
- Do not modify v1 artifacts: `notes/` (v1 guide, inputs, parts, batches), `src/lifemate/notes.jsonl`. v2 lives in
  `notes_v2/` and `src/lifemate/notes_v2.jsonl`.
- No Kaggle pushes, dataset uploads, git commits or pushes. Preparing a plan file is fine; pushing it is not.
- Run Python that needs scikit-learn or scipy with a Python environment that has them.
  Local data: `LIFEMATE_DATA=$PWD/train+public_test/train+public_test`. Tests:
  `LIFEMATE_DATA=$PWD/train+public_test/train+public_test python -m pytest -q tests` (53 pass now; keep them passing).

## 3. The v2 note rules

**R1. State the rule, not the outcome.** A note may use only what the model sees before the first call of the final
turn (time, profile, history with its tool results, the request). Whenever the right action depends on what a read of
this turn returns, the note names the read and states the decision rule with **both branches**, worded the same way
whether the key ends up acting or asking. Sort every sample into one kind with a script, from its gold calls:

| Kind | When | Note shape (Vietnamese, adapt the words) |
|---|---|---|
| K0 explicit | no read of this turn decides anything (no read, or the plan is fixed whatever the reads return) | as v1: resolved dates and times with the arithmetic, counts, references to earlier turns, the tools in order |
| K1 pick one item | a read must identify one item (an ID the key takes from the read, or the key asks for such an ID after a read) | "`<read>` tìm <mô tả>; khớp đúng một thì `<write tool>`, không có hoặc nhiều thì sys.ask_missing_info cho `<write tool>`, params `<param>`" |
| K2 condition on a read | the request has a condition that a read of this turn decides (weather, balance, battery, an alert, an SMS) | "`<read>` xem <điều kiện>; <đúng> thì `<tools>`, không thì <thôi hoặc tools khác>" |
| K3 select by a rule | a read lists items and the action applies to all that match a rule | "`<read>`; `<write tool>` từng mục <quy tắc chọn>" — never the count or the IDs |
| K4 value lookup | a read only supplies a value for an action that happens anyway (a bill amount, today's steps, which account has enough) | "`<read>` lấy <giá trị>, rồi `<write tool>` với <giá trị đó / công thức>"; if/else on the value is a rule, fine |

A sample with several sub-requests can mix kinds; each part follows its own shape. Forbidden in K1–K3 notes: anything
that reveals the branch (e.g. "không xác định được", "chỉ đọc", "không có nên thôi", "vẫn còn", "đã có"), a count or ID
that only the read shows, and committing to the write without the other branch.

**Branch-blind inputs (how to make R1 hold by construction).** For K1, K2 and K3 samples the writer must not be able
to tell which branch the key took. Build the writer's input from the request, history and the gold calls up to and
including the deciding read(s); replace everything after with the same neutral line for both outcomes (K1: the write
tool and the parameter the read must supply; K2/K3: "depends on the read result", plus the installed tools with their
descriptions so the writer can name the branch tools). `must_include` for these kinds comes from the request, history
and the read part only, never from the write's arguments. For K0 and K4, inputs are as in v1.

**R2. No invented details.** Every date, time, number and ID in a note must be a gold argument value of the final
turn, or a step of an explicit arithmetic chain that ends in one (`13:10 + 45 phút = 13:55`), or (K1–K3 only) the
conversion of a value spoken in the request for the intended action (`năm mươi ngàn = 50000`). Do not mention the
current time or date unless it feeds such a chain; do not mention optional parameters or filters the key does not set
(`near`, `merchant`, `delay_minutes`, `label`, a workout `time`); write search terms exactly as the key's query; when a
tool name could suggest a different enum value, state the key's value (`email.archive action delete`).

**R3. Keep what worked.** Time and date arithmetic, counts, what "đó / ổng / bữa đó" refers to in the history (with
its ID when the history shows it), all sub-tasks of a multi-step request in the key's order, a 2–5 word reason for an
unfamiliar tool taken from its description, the missing parameter for an ask that is knowable before any read, and
"không gọi tool" with a reason for an empty answer.

**R4. v1 limits stay.** One line, 10–35 words (hard limits 60 words, 100 Qwen tokens), no `< > { }`, no JSON, no
restating the request, names the first gold tool, names only installed tools (plus sys.ask_missing_info).

## 4. Work plan

1. **Kinds.** `scripts/notes/notes_v2_common.py`: classify the 1,159 non-validation train samples into K0–K4 from
   gold calls, tool kinds (catalog `kind`), `x-ref` parameters, which values are hidden (reuse `notes_common.py`), and
   the request text for conditions. Print counts by batch (A = ids < 425, B < 919, C otherwise) and kind; expect about
   800 in K1–K4. Review 10 random samples per kind by hand and fix the rules until the kinds look right; record the
   final counts in `log.md`.
2. **Inputs.** `scripts/notes/make_inputs_v2.py` → `notes_v2/inputs/batch_NNN_pK.txt`, 50 samples per batch, 10 per
   part, shuffled with seed 2026, branch-blind for K1–K3. Assert no validation id anywhere.
3. **Guide.** `notes_v2/GUIDE.md`, built from `notes/GUIDE.md` plus R1–R3, with one good and one bad example per kind
   (from non-validation train samples or made up). The writer reads only this guide and its input part.
4. **Checker.** `scripts/notes/check_notes_v2.py` = the v1 checks with v2 `must_include`, plus: R1 shape for K1–K3
   (both branches present, the decision tool and parameter named for K1, no branch-revealing phrase from a list you
   keep in the script), R2 (no value outside the allowed sources, no parameter name the key does not set for the tools
   the note names). Add tests in `tests/test_notes.py` with made-up strings for every rule.
5. **Carry over or rewrite.** Run the v2 checker on the v1 notes. Keep a v1 note unchanged only if it is K0 and passes
   v2; everything else is rewritten. Report the counts.
6. **Pilot, then write.** Pilot on 20 samples that cover every kind; continue only if at least 18 pass the checker and
   your own reading finds no outcome leak; otherwise fix the guide or inputs first. Then write the rest with writer
   subagents (Agent tool, model `sonnet`), one per input part, at most four at a time. Each writer is told to read
   only `notes_v2/GUIDE.md` and its input part and to write only its output file `notes_v2/parts/<part>.jsonl`.
   Assemble each batch to `notes_v2/batch_NNN.jsonl`, run the checker, send failures back for at most two fix rounds,
   then drop a note with its reason (dropping more than 2 % means the guide or checker needs fixing, not more drops).
7. **Leak probe (the acceptance test for R1).** `scripts/notes/leak_probe.py`: on K1 samples, predict "key asks" vs
   "key acts"; on K2 samples, predict "key writes" vs "key only reads". Logistic regression on word 1–2-gram TF-IDF,
   5×5 repeated stratified CV, AUC with its spread, for (a) request + history text alone and (b) the same plus the note.
   The note must not add information: AUC(b) − AUC(a) ≤ 0.02 for v2. Run the same probe on the v1 notes as a positive
   control; it must show a clear gain there, otherwise strengthen the probe before trusting it. Print the note-only
   n-grams with the largest weights and fix any that reveal the branch.
8. **Package.** `src/lifemate/notes_v2.jsonl` (built like `build_package_notes.py`); `train.py --notes package-v2`
   loads it (keep `package` = v1), with a test. Prepare, do not push, `scripts/plans/notes_v2_folds.json`: a copy of
   `scripts/plans/notes_folds.json` with `--notes package-v2` and the output names `n1`/`n2`/`notes_eval` changed to
   `m1`/`m2`/`notes2_eval`.

## 5. Done means all of these hold

- Every one of the 1,159 non-validation samples has a v2 note that passes `check_notes_v2.py`, or is dropped with a
  stated reason (at most 2 %).
- No validation id appears in any v2 input, part, batch, guide or package file (asserted by the scripts).
- Leak probe: v2 adds at most 0.02 AUC for both K1 and K2; v1 shows a clear gain; numbers in `log.md`.
- Zero occurrences of the v1 branch phrases in K1–K3 notes.
- Tests pass. v1 files unchanged (`git diff --stat notes/ src/lifemate/notes.jsonl` is empty).
- A `log.md` section "Worked notes v2" with: kind counts by batch, carried-over vs rewritten counts, checker pass rates,
  drops with reasons, probe results (v1 and v2), five before/after note pairs from training samples, subagent token
  use, and wall time.

## 6. Stop and report

Stop when everything in section 5 holds, or after about 8 hours of work, or if you are blocked (a rule you cannot
satisfy, a checker you cannot make reliable, a probe that will not separate v1). Then tell the user, in a few plain
sentences: what was done, the probe result, whether the notes are ready for a validation run, and the exact commands
that would launch it on Kaggle. The run needs about 9.5 GPU hours; say which account has that much left
(`python scripts/kg.py --account <account> quota`), and that on another account the datasets named below would first
have to be copied there. Commands: first the code upload (`python scripts/kg.py --account <account> code`, needed because
`train.py` and `notes_v2.jsonl` changed), then the push, modelled on the `notes-folds` command in `log.md`
(`kg.py --account <account> push --hours 11.5 --dataset OWNER/lifemate-asr-exp7 --dataset
OWNER/lifemate-avg5-transcripts --dataset OWNER/lifemate-soup-llm notes2-folds jobs -- --plan "$(tr -d
'\n' < scripts/plans/notes_v2_folds.json)" --deadline-hours 11.2`). Run neither.
