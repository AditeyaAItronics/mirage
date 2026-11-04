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

### 3. ✅ DONE (Nov 2026, D34): Start the DistilBERT smoke test, then the real run (2 min of clicking, ~1–2 h of waiting)
- First run with `SMOKE_TEST = True` and `SEEDS = [42]` (about 5 minutes; it skips the IMDb
  predictions) to catch setup problems cheaply.
- Cell 2 prints `code version:`. Check that it shows the latest commit on GitHub. Re-running in
  the same session is safe: it updates an existing clone instead of reusing the old code.
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
`prepare`, `train-classical --variants base negmark negmark_w3 unigram` and the whole classical pipeline (`all`) have run;
see `majordecisions.md` D26 for the first results. Re-run them only if you change data or model configs:
```bash
.venv/Scripts/python -m mirage prepare
.venv/Scripts/python -m mirage train-classical --variants base negmark negmark_w3 unigram
```
- `prepare` downloads SST-2 (with public test labels) and IMDb (~100 MB) and prints `results/data_stats.csv`.
  Current sizes: train 63,971 / held-out 3,378 / val 872 / **test 1,820** / **negation 519** / IMDb 10,000.
- `train-classical` takes minutes for most models. **The random forest is the slow one.** If RAM or
  time becomes a problem, see "Decisions" below.

---

## Week 1 — decisions only you can make

### 5. ✅ DONE (Nov 2026): Fix the factual statements in the proposal
Applied to `Downloads/Mirage_Full_Proposal_4_Week.md`; the untouched original is saved next to it
as `Mirage_Full_Proposal_4_Week.original.md`. Besides the three below, these were corrected to
match the code:
- test set (1,820 SST-2 test sentences; validation is for choices only);
- the leak-free held-out split;
- IMDb (10,000 reviews, capped, 512 tokens with start + end kept);
- the tuning grids for LR and the decision tree;
- DistilBERT's seeds and median-seed rule;
- the full behavioral suite table (with the name-shortcut test explained);
- the second negation-scope rule;
- the per-sentence early-warning score with the length baseline;
- Holm correction.

**✅ Resolved (D42):** H2 says "single-word features" and H3 "word-pair features … close part of
the gap", but every classical model used words *and* word pairs. A single-word variant now
completes the ladder, so H2 and H3 are tested exactly as worded, and the hypotheses table stays
unchanged.

The three original corrections (all applied):
1. "Real negation subset = SST-2 **test** sentences" → it is now validation **plus** the public SST-2 test sentences (519 in total). GLUE hides test labels, so the test labels come from the original Stanford release via `SetFit/sst2` (D22). Say so.
2. Negation tests: say that *"not + negative adjective"* cases are reported separately and not scored (no clean expected label).
3. Behavioral test count: now **3,750 cases = 6,250 rows** (invariance and directional tests use pairs; the directional "X / not X" test (D27), the name-shortcut test (D32) and the noun-typo control (D37) are new). IMDb sample: **10,000** reviews, each capped to its first 220 + last 140 words for every model (D23).

### 6. ✅ No action needed: the random forest trained fine (about 4 min per fit); only its files are large
It trained fine here (about 4 min per fit), but saves at **822 MB** per variant. If disk or RAM is tight:
- Keep the grid, but train it alone overnight: `python -m mirage train-classical --models rf --variants base negmark negmark_w3`
- Drop `null` from `configs/classical.yaml → rf.grid.max_depth` (keep `[30]`), and say so in the report.
- Reduce `n_estimators` to 100.

### 7. Review the behavioral word lists — ✅ done ("messy" → "sloppy", everything else kept; D33)
`src/mirage/behavioral/wordlists.py` and the `name_pairs` in `configs/behavioral.yaml`. You know your audience. Also check that the name pairs (Kidman↔Seagal, Spielberg↔Costner, …, chosen from SST-2's training labels, evidence in `results/name_associations.csv`) and the mild adjectives (`MILD_POS` / `MILD_NEG`) are ones you're happy to defend. Check that every adjective is clearly
positive or negative *in a movie review*, and that the names are a fair cross-section (the
name-swap test uses Indian vs Western names). Then run `python -m mirage build-suite` and skim
`results/behavioral_suite.csv`. Skimming the 6,250 short lines (or `results/word_errors.csv`) is faster for you than explaining which
ones feel unnatural.

---

## Now — the presentation additions (D39–D41)

### 14. ✅ DONE (Nov 2026, D46): Run the external-models notebook on Kaggle (about 20–30 min of GPU, 5 min of clicking)
This gives RoBERTa-large and Hugging Face's DistilBERT predictions on **every** evaluation set,
including all 10,000 IMDb reviews, so they're compared with our models on exactly the same data.
- New Notebook → *File → Import notebook* → Link →
  `https://raw.githubusercontent.com/AditeyaAItronics/mirage/main/notebooks/kaggle_external.ipynb`
  (import fresh: an older imported copy runs old code).
- Settings: **GPU T4** (x1 is enough), **Internet = On**.
- *Save Version → Save & Run All*. No settings to change in the notebook.
- Check cell 2's `code version:`: it should be the latest commit on GitHub. Cell 3 prints the
  number of IMDb reviews (10,000) and the first ids (`imdb-0`, `imdb-1`, `imdb-2`, as on the laptop).
- When it finishes: Output → download `external_outputs.zip` → unzip into the **repo root**. You
  should get 14 files: `results/predictions/{roberta_large,distilbert_hf}__base__seed0__*.parquet`.
- Then tell Claude. It checks that the reviews match the laptop's, re-runs `evaluate`, `warn` and
  `figures` (every table and figure gains the two models), and writes up the results.
- **What you'll see, and what's normal** (from the first run, Nov 2026):
  - **Internet must be On** in the notebook's settings. It can switch off when a session
    restarts, and without it the run fails within the first minute (it can't clone, install or
    download the models). That is what stopped the first attempt.
  - **"Debugger warning: frozen modules…"** at the very start: harmless, Kaggle prints it in
    every notebook.
  - **"Token indices sequence length is longer than … (535 > 512)"** around 3 minutes: harmless.
    The full review is tokenized first, then cut to 512 tokens (start + end kept, D23) before the
    model sees it.
  - **A long quiet stretch** after that: RoBERTa-large working through the 10,000 IMDb reviews.
  - **You can close the laptop.** A saved version runs on Kaggle's servers; only pressing Run All
    in the editor (without saving a version) depends on your session.
  - The version badge shows **1** when it has finished; open that version to download the output.

### 15. Try the live demo yourself (10 min)
```bash
.venv/Scripts/python -m pip install -e ".[dl,demo]"
```
```bash
.venv/Scripts/python -m streamlit run app/shortcut_explorer.py
```
It opens at http://localhost:8501. Read `app/PRESENTING.md` (the demo script and ready lines for
when a sentence goes wrong), click through the curated examples, then try your own sentences.
**Your call:** are the curated examples ones you're happy to show? Their wording is in
`app/examples.yaml`; if you change one, run `python -m pytest tests/test_demo.py` to re-check its
expected behaviour, and `python app/make_fallbacks.py` to redraw the fallback images.

### 16. Look at the two new figures
`figures/mirage_gap_slide.png` (the headline: accuracy vs robustness) and
`figures/shortcut_fingerprint_slide.png`. Also `figures/negation_gap_sst2_neg.png` (the H2/H3
ladder, D42) and `figures/h5_word_effect_agreement.png` (H5, D43). The robustness score's definition is in D40. Check that
you can explain it in one sentence: *"the share of shortcut tests a model gets fully right"*.

---

## Weeks 2–3 — run, then read the results with your own eyes

### 8. ✅ DONE (D35): After the DistilBERT zip is unzipped
```bash
.venv/Scripts/python -m mirage all
```
This builds the suite and then runs predict → evaluate → explain (incl. IG) → warn (embeddings) → figures.
The first `warn` run embeds 67k sentences on CPU. Expect several minutes; it is cached after that.

### 9. ✅ Checked by Claude (D35); a look from you is still worthwhile: Sanity-check numbers before trusting them
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

### 12. Write the report narrative — follow [reportplan.md](reportplan.md)
The plan lists every step in order (yours and Claude's), the report skeleton with the source of
each section, and the hypothesis verdicts to approve. **To start, send Claude: the format rules,
"verdicts approved" (or your changes), and your main-text / appendix choice.**
The source material is ready: [hypothesisanswers.md](hypothesisanswers.md) (H1–H6 in full) and
[interestingfindings.md](interestingfindings.md) (13 further findings, each tied to a hypothesis).
Pick from them what goes into the report and how much.
Claude can draft sections, tables and figure captions from the CSVs (ask for a Word/PDF report via
the docx or pdf skills). But **the argument, the verdict for each hypothesis, and the
limitations section are yours**: you are graded on them, and you know what your evaluators expect.

### 13. Final checks before submission
- `python -m pytest`: all green.
- Ask Claude to run `/security-review` before making the repo public (it catches stray tokens).
- ~~Fill in the README "Results" section with the main table and one figure.~~ Done; add the
  external models' rows once their Kaggle run is in.

---

## Things Claude can keep doing for you (just ask)

- Fix anything that breaks on real data. Paste the error, or let Claude run the stage.
- Run `/code-review` at the end of each week on that week's changes.
- Add the Lee et al. class-conditional variant (already implemented: `GaussianDensity(class_conditional=True)`) to the `warn` stage as a second distance.
- Build the stretch goal (negation-augmented DistilBERT) on its own branch.
- Turn the CSVs into the report's tables, and the final report into a .docx or PDF.
