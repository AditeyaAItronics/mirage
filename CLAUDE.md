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
  Exception: the ready-made external models (`models/external.py`, D39) read raw text unnormalized.
- **Additions never change earlier statistics.** Every multi-model Holm family is split with
  `evaluate.holm_family`: ours / external models (D39) / ablation variants (`unigram`, D42).
  H2's `negation_gap` excludes external models, and `negmark_effect` excludes ablations. After
  adding models or variants, re-run `evaluate` and check the committed CSVs' existing rows are
  unchanged.
- **Classical variants:** `base` (words + word pairs), `negmark` / `negmark_w3` (+ negation
  marking), `unigram` (single words only, the first rung of the H2/H3 ladder). Ablation variant
  names live in `models.base.ABLATION_VARIANTS`, so analysis code never imports model classes.
- **Don't edit `src/` while a stage is running:** lazy imports can mix old and new modules mid-run.
- **Held-out is split by sentence** (`data.heldout_mask`): never replace it with a random row split. SST-2 rows are sub-phrases, so a random split leaks (D21).
- **Never use GLUE's SST-2 test labels** (they are `-1`). The test set comes from `SetFit/sst2` (original public Stanford labels, D22).
  `sst2_val` is for choices (DistilBERT's median seed); `sst2_test` is the reported in-distribution set. The negation subset pools both.
- **IMDb text is capped** to head + tail words in `prepare`, so every model reads the same text (D23). Don't feed raw reviews to one model only.
- **No stop-word removal** and **no default sklearn `token_pattern`** — they delete `not` / `n't`.
  Vectorizers use `tokenizer=str.split, token_pattern=None, lowercase=False`.
- Heavy libraries (`torch`, `transformers`, `datasets`, `sentence-transformers`, `captum`) are imported
  inside functions only.
- All paths come from `mirage.paths`; all settings from `configs/*.yaml`.
- Log every major design change in `majordecisions.md`.

## Keep the docs current as you build and fix
Update every relevant markdown file **as part of the same change**, not at the end:
- `majordecisions.md`: every decision taken (what, why, what was rejected, the numbers), as it is made.
- `README.md`: new or changed results, figures and commands.
- `ARCHITECTURE.md`: new stages, modules, interfaces, layering or data-flow changes.
- `shortfalls.md`: new limitations found; move fixed ones to Resolved.
- `newadditionsdesign.md`: what has been done, what hasn't, and any fixes or changes to the design.
- `codeexplanations.md`: new or changed files (and the test count); `whatyouneedtodo.md`: steps for the user.
- `CLAUDE.md`: new hard rules.

## Commands (Windows, from repo root)
- Tests: `.venv/Scripts/python -m pytest`
- Lint/format: `.venv/Scripts/python -m ruff check src tests app` / `ruff format src tests app`
- Stages: `.venv/Scripts/python -m mirage <stage>` (`python -m mirage --help` lists them)
- Demo: `.venv/Scripts/python -m streamlit run app/shortcut_explorer.py` (needs the `[demo]` extra); the app is layout only, its logic is `mirage.demo`

## Seeds
Data/classical/behavioral: 42. DistilBERT: 13, 42, 87 (per-item analyses use the median-accuracy seed). External models: 0 (no training).
