# What you need to do

The code skeleton is built and tested. Everything below is either **something only you can do**
(accounts, GPU, decisions, academic judgment) or **something you can do faster than Claude**
(clicking in web UIs, waiting on downloads, reading your own results with domain knowledge).
Ordered by when you need it.

---

## Today (≈ 30 minutes) — unblock everything else

### 1. Put the repo on GitHub — ✅ done
The code is pushed to https://github.com/AditeyaAItronics/mirage (branch `main`), and the notebook's `REPO_URL` points there.
**Keep the repo public** while Kaggle runs. If it's private, the notebook's `git clone` needs a token.
After you change code locally: `git add -A && git commit -m "..." && git push`.

### 2. Kaggle account + phone verification (10 min) — *only you can*
- Sign up at kaggle.com and **verify your phone number**. Without it you get no GPU and no internet in notebooks.
- New Notebook → *File → Import notebook* → `notebooks/kaggle_distilbert.ipynb`.
- Settings panel: **Accelerator = GPU T4 x1**, **Internet = On**.
- `REPO_URL` is already set to the GitHub repo.

### 3. Start the DistilBERT smoke test, then the real run (2 min of clicking, ~1–2 h of waiting)
- First run with `SMOKE_TEST = True` (≈ 2 min) to catch setup problems cheaply.
- Then `SMOKE_TEST = False`, *Save Version → Save & Run All (Commit)*. A committed run keeps going
  after you close the tab. Free quota is about 30 GPU-hours a week; 3 seeds of training plus GPU predictions use about 1–2 h.
- When it finishes: Output tab → download `distilbert_outputs.zip` → unzip it into the **repo root**
  (`C:\claude\Mirage`). You should get `artifacts/models/distilbert__base__seed13/42/87` and
  `results/predictions/distilbert__*.parquet`. The notebook makes DistilBERT's predictions on the
  GPU, because 10,000 IMDb reviews at 512 tokens × 3 seeds is slow on a laptop. The local `predict`
  stage skips models whose predictions are newer than the model files, so it won't redo them.
- **Use a run started after commit "Scale up evaluation…"** (Nov 2026). Older runs used the leaky
  held-out split and the 256-token IMDb cut.

*Why you're faster:* Claude can't log in to Kaggle, accept its terms, or verify a phone. A committed
Kaggle run needs nobody watching it.

### 4. Data download and classical training: ✅ already done on this laptop (Nov 2026)
`prepare`, `train-classical --variants base negmark negmark_w3` and the whole classical pipeline (`all`) have run;
see `majordecisions.md` D26 for the first results. Re-run them only if you change data or model configs:
```bash
.venv/Scripts/python -m mirage prepare
.venv/Scripts/python -m mirage train-classical --variants base negmark negmark_w3
```
- `prepare` downloads SST-2 (with public test labels) and IMDb (~100 MB) and prints `results/data_stats.csv`.
  Current sizes: train 63,971 / held-out 3,378 / val 872 / **test 1,820** / **negation 519** / IMDb 10,000.
- `train-classical` takes minutes for most models. **The random forest is the slow one.** If RAM or
  time becomes a problem, see "Decisions" below.

---

## Week 1 — decisions only you can make

### 5. Fix three statements in the proposal before you submit or present it
These are factual corrections, and they are your academic call:
1. "Real negation subset = SST-2 **test** sentences" → it is now validation **plus** the public SST-2 test sentences (519 in total). GLUE hides test labels, so the test labels come from the original Stanford release via `SetFit/sst2` (D22). Say so.
2. Negation tests: say that *"not + negative adjective"* cases are reported separately and not scored (no clean expected label).
3. Behavioral test count: now **2,750 cases = 4,250 rows** (invariance and directional tests use pairs; the directional "X / not X" test is new, D27). IMDb sample: **10,000** reviews, each capped to its first 220 + last 140 words for every model (D23).

### 6. Decide on the random forest if it is too heavy
It trained fine here (about 4 min per fit), but saves at **822 MB** per variant. If disk or RAM is tight:
- Keep the grid, but train it alone overnight: `python -m mirage train-classical --models rf --variants base negmark negmark_w3`
- Drop `null` from `configs/classical.yaml → rf.grid.max_depth` (keep `[30]`), and say so in the report.
- Reduce `n_estimators` to 100.

### 7. Review the behavioral word lists (10 min) — *you're faster*
`src/mirage/behavioral/wordlists.py`. You know your audience. Check that every adjective is clearly
positive or negative *in a movie review*, and that the names are a fair cross-section (the
name-swap test uses Indian vs Western names). Then run `python -m mirage build-suite` and skim
`results/behavioral_suite.csv`. Skimming the 4,250 short lines (or `results/word_errors.csv`) is faster for you than explaining which
ones feel unnatural.

---

## Weeks 2–3 — run, then read the results with your own eyes

### 8. After the DistilBERT zip is unzipped
```bash
.venv/Scripts/python -m mirage all
```
This builds the suite and then runs predict → evaluate → explain (incl. IG) → warn (embeddings) → figures.
The first `warn` run embeds 67k sentences on CPU. Expect several minutes; it is cached after that.

### 9. Sanity-check numbers before trusting them — *you're faster at "does this look right?"*
| Check | Expect | If not |
|---|---|---|
| DistilBERT SST-2 val accuracy | ≈ 0.90–0.91 | training problem: check `train_summary.json` |
| LR SST-2 val accuracy | ≈ 0.80–0.83 | vectorizer problem |
| GNB below MNB | likely (H6) | still a finding |
| `ood_sanity_auroc` in `early_warning.csv` | close to 1.0 | embedding or distance bug |
| IG max convergence delta (printed) | < 0.05 | raise `n_steps` in `explain/ig.py` |

### 10. Choose the 8–10 failure examples and write their explanations — *only you can*
Use `results/explain/ig_failures.csv`, `top_features.csv` and `tree_depth4.txt`. Choosing *which*
failures tell the story, and explaining the shortcut in plain words, is the core of the report and
is your judgment.

### 11. Stretch goal go/no-go (end of week 3) — *your call*
Only if the week 3 checkpoint is met early. Ask Claude to build it on a separate git worktree.

---

## Week 4 — the report

### 12. Write the report narrative
Claude can draft sections, tables and figure captions from the CSVs (ask for a Word/PDF report via
the docx or pdf skills). But **the argument, the verdict for each hypothesis, and the
limitations section are yours**: you are graded on them, and you know what your evaluators expect.

### 13. Final checks before submission
- `python -m pytest`: all green.
- Ask Claude to run `/security-review` before making the repo public (it catches stray tokens).
- Fill in the README "Results" section with the main table and one figure.

---

## Things Claude can keep doing for you (just ask)

- Fix anything that breaks on real data. Paste the error, or let Claude run the stage.
- Run `/code-review` at the end of each week on that week's changes.
- Add the Lee et al. class-conditional variant (already implemented: `GaussianDensity(class_conditional=True)`) to the `warn` stage as a second distance.
- Build the stretch goal (negation-augmented DistilBERT) on its own branch.
- Turn the CSVs into the report's tables, and the final report into a .docx or PDF.
