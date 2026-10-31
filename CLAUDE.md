# Mirage — notes for Claude Code

Research codebase: shortcut learning in sentiment classifiers (SST-2 → IMDb).
Read `ARCHITECTURE.md` for the design, `majordecisions.md` for why, and
`MIRAGE_IMPLEMENTATION_PLAN.md` for the week-by-week build order.

## Hard rules
- **Predictions are the contract.** Every model writes `results/predictions/{model}__{variant}__seed{seed}__{dataset}.parquet`
  via `mirage.predict.write_predictions`. Metrics, stats, plots and early warning read only these files.
  Validate with `mirage.predict.validate_predictions`.
- **Raw text in, normalization inside.** Classifiers receive raw text; `mirage.text.normalize`
  (and `negation_mark` for the `negmark` variant) run inside the model pipeline. Never normalize twice.
- **Never use SST-2 test labels** (they are `-1`). The negation subset comes from validation.
- **No stop-word removal** and **no default sklearn `token_pattern`** — they delete `not` / `n't`.
  Vectorizers use `tokenizer=str.split, token_pattern=None, lowercase=False`.
- Heavy libraries (`torch`, `transformers`, `datasets`, `sentence-transformers`, `captum`) are imported
  inside functions only.
- All paths come from `mirage.paths`; all settings from `configs/*.yaml`.
- Log every major design change in `majordecisions.md`.

## Commands (Windows, from repo root)
- Tests: `.venv/Scripts/python -m pytest`
- Lint/format: `.venv/Scripts/python -m ruff check src tests` / `ruff format src tests`
- Stages: `.venv/Scripts/python -m mirage <stage>` (`python -m mirage --help` lists them)

## Seeds
Data/classical/behavioral: 42. DistilBERT: 13, 42, 87 (per-item analyses use the median-accuracy seed).
