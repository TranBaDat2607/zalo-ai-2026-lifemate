# DataMate solver

Rule-based question answering over the shop data packages. No language model is used at
inference, so the solution runs offline, is deterministic, and uses 0 model parameters.

## Layout

| File | Role |
|---|---|
| `values.py` | parsing of noisy numbers (`84,000`, `2,76tr`, `12tr460`, `105k`, `3 ly`) and dates (`1 thg 3 2025`, Excel serials, `1/3`) |
| `tables.py` | reading csv / xlsx (python-calamine) / sqlite, header detection, footer and subtotal removal |
| `readme.py`, `schema.py` | README parsing; README column descriptions and column-name aliases mapped to canonical fields |
| `loader*.py` | canonical tables per shop: orders, lines, products, units, payments, purchases, stock, prices, lots, deliveries, due dates, expenses; special books (repair shop, phone shop, invoice voucher lines) |
| `periods.py` | time expressions (rulebook §2), lunar calendar for Tết |
| `qa.py` | question normalisation (accents, Telex typos, abbreviations, dialect synonyms), entity matching, dispatcher |
| `handlers*.py` | intents: revenue / orders / AOV / quantity with filters, comparisons, shares, rankings, thresholds, debt, stock, purchases, profit, lots and expiry, price history, deliveries, domain rules (§6) |
| `run.py` | writes `submission.jsonl` and `logs/answers_log.jsonl` (id, final answer, handler, elapsed time) |
| `evaluate.py`, `scoring.py` | train-split evaluation with the competition matching rules |

## Usage

```
pip install pandas numpy python-calamine
python solver/run.py --data data/public_test --out output/submission.jsonl --logs output/logs
python solver/evaluate.py            # all train businesses
python solver/evaluate.py biz_117    # one business
```

Parsed shop data is cached under `.cache/` (or `$DATAMATE_CACHE`), keyed on the files' modification time
and `LOADER_VERSION`.

Questions the solver cannot route return `abstain` (score 0); questions whose period lies outside the
data or that need information the shop does not record return `unanswerable`.

## Models

None. Total parameter count: 0.
