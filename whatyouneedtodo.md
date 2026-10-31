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

### 3. Start the DistilBERT smoke test, then the real run (2 min of clicking, ~1.5 h of waiting)
- First run with `SMOKE_TEST = True` (≈ 2 min) to catch setup problems cheaply.
- Then `SMOKE_TEST = False`, *Save Version → Save & Run All (Commit)*. A committed run keeps going
  after you close the tab. Free quota is about 30 GPU-hours a week; 3 seeds use about 1.5 h.
- When it finishes: Output tab → download `distilbert_models.zip` → unzip it into
  `artifacts/models/` here. You should get three folders, `distilbert__base__seed13/42/87`.

*Why you're faster:* Claude can't log in to Kaggle, accept its terms, or verify a phone. A committed
Kaggle run needs nobody watching it.

### 4. Run the data download and classical training locally (start it, then walk away)
```bash
.venv/Scripts/python -m mirage prepare
.venv/Scripts/python -m mirage train-classical --variants base negmark
```
- `prepare` downloads SST-2 and IMDb (~100 MB) and prints `results/data_stats.csv`.
  **Write down the size of `sst2_neg`.** The proposal says "size measured in week 1", and the report needs it.
- `train-classical` takes minutes for most models. **The random forest is the slow one.** If RAM or
  time becomes a problem, see "Decisions" below.

---

## Week 1 — decisions only you can make

### 5. Fix three statements in the proposal before you submit or present it
These are factual corrections, and they are your academic call:
1. "Real negation subset = SST-2 **test** sentences" → it must be the **validation** set (test labels are hidden). The code already does this.
2. Negation tests: say that *"not + negative adjective"* cases are reported separately and not scored (no clean expected label).
3. Behavioral test count: **400 cases = 550 rows** (invariance tests use pairs). The proposal's per-type numbers still hold.

### 6. Decide on the random forest if it is too heavy
Full-depth trees × 300 on ~64k rows can take a few GB of RAM. Options, in order of preference:
- Keep the grid, but train it alone overnight: `python -m mirage train-classical --models rf --variants base negmark`
- Drop `null` from `configs/classical.yaml → rf.grid.max_depth` (keep `[30]`), and say so in the report.
- Reduce `n_estimators` to 100.

### 7. Review the behavioral word lists (10 min) — *you're faster*
`src/mirage/behavioral/wordlists.py`. You know your audience. Check that every adjective is clearly
positive or negative *in a movie review*, and that the names are a fair cross-section (the
name-swap test uses Indian vs Western names). Then run `python -m mirage build-suite` and skim
`results/behavioral_suite.csv`. Reading 550 short lines is faster for you than explaining which
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
