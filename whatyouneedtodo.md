# What you need to do

Every step for you, in order, including the report's. Each one is either **something only you
can do** (decisions, academic judgment, accounts) or **something you can do faster than Claude**.
The code, the experiments and the analysis are finished (see "Done" at the end).

Where to find things:
- **Results:** [hypothesisanswers.md](hypothesisanswers.md) (H1–H6 in full) and
  [interestingfindings.md](interestingfindings.md) (13 further findings).
- **The report's skeleton and sources:** [reportplan.md](reportplan.md).
- **Why each choice was made:** [majordecisions.md](majordecisions.md); **limitations:**
  [shortfalls.md](shortfalls.md).

---

## Now

### 1. Try the live demo yourself (10 min)
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

### 2. Look at the main figures (10 min)
`figures/mirage_gap_slide.png` (the headline: accuracy vs robustness),
`figures/shortcut_fingerprint_slide.png`, `figures/negation_gap_sst2_neg.png` (the H2/H3 ladder)
and `figures/h5_word_effect_agreement.png` (H5). Check that you can explain the robustness score in
one sentence: *"the share of shortcut tests a model gets fully right"* (D40).

### 3. Read the two results files (1 h)
[hypothesisanswers.md](hypothesisanswers.md) and [interestingfindings.md](interestingfindings.md)
are deliberately longer than the report. Mark what you want in the report and how much; anything
unclear or that you'd word differently, tell Claude.

---

## The report

What happens, in order. "You" steps need your judgment; Claude does the rest when you ask.

| Step | Who | What | Time |
|---|---|---|---|
| 4 | **You** | **Tell Claude the format rules:** page limit, required template (Word / LaTeX / course template), citation style, required sections (e.g. ethics statement, contribution statement), due date, and whether appendices count toward the limit. | 5 min |
| 5 | **You** | **Approve the hypothesis verdicts.** Each hypothesis's "Verdict" section in [hypothesisanswers.md](hypothesisanswers.md) is the proposed wording (summary table at the top). How strongly to word each verdict is your academic call. | 15 min |
| 6 | **You** | **Decide what goes in the main text and what in an appendix:** which of the 13 findings, and the extras (the external models D46, the Mirage Gap D40, the name probe D45, the live demo D41). [reportplan.md](reportplan.md) §6 has a suggestion. | 10 min |
| 7 | Claude | Shortlist about 20 failure-example candidates in `report/failure_candidates.md`. For each: the sentence, what each model said, and which shortcut it shows. Sources include `results/explain/ig_failures.csv`, `top_features.csv`, `tree_depth4.txt` and the behavioral predictions. | — |
| 8 | **You** | **Pick 8–10 of them** and approve (or rewrite) the one-line explanation of each. Choosing the examples that tell the story, and explaining the shortcut in plain words, is the core of the report; you'll be asked about them. | 30 min |
| 9 | Claude | Draft the report section by section, following [reportplan.md](reportplan.md), in the agreed format, with every figure and table and captions that state the result. | — |
| 10 | Claude | **Number check:** a script compares every number in the draft with `results/*.csv` and flags any mismatch. | — |
| 11 | **You** | **Review the draft.** Read it as the examiner would; mark anything unclear, overstated or not in your voice. Claude revises. One or two rounds. **The argument, the verdicts and the limitations are yours**: you are graded on them. | 1–2 h |
| 12 | **You** | **Write or approve the personal parts:** abstract wording, acknowledgements, anything the course requires in your own words. | 30 min |
| 13 | Claude | Final pass: references, figure numbering, page count, export (Word / PDF). | — |
| 14 | **You** | Submit. | — |

**To start, send Claude one message with steps 4–6:** the format rules, "verdicts approved" (or
your changes), and your main-text / appendix choice. Steps 7 and 9 can then run straight away.

**Presentation slides** are a separate deliverable. They can reuse the report's figures; ask when
the report draft is done.

### 15. Final checks before submission
- `python -m pytest`: all green.
- Ask Claude to run `/security-review` before submitting the repo link (it catches stray tokens).
- Keep the repo public if the examiners need to see it.

---

## Optional

### 16. Stretch goal — *your call*
A negation-augmented DistilBERT (the proposal's stretch goal). Only if there is time before the
report; ask Claude to build it on its own branch.

### Things Claude can do for you (just ask)
- Fix anything that breaks. Paste the error, or let Claude run the stage.
- Run `/code-review` on recent changes.
- Add the Lee et al. class-conditional variant (already implemented:
  `GaussianDensity(class_conditional=True)`) to the `warn` stage as a second distance.
- Address the remaining 🟠 shortfalls (B10, C3, C4, E1, F2 in [shortfalls.md](shortfalls.md)).
- Turn the CSVs into the report's tables, and the final report into a .docx or PDF.
- Put the demo online (Hugging Face Spaces, D48) if you want a public link.

---

## Done

Kept as a record; nothing to do here unless you re-run something.

| Step | Done |
|---|---|
| Repo on GitHub | https://github.com/AditeyaAItronics/mirage (branch `main`). After local changes: `git add -A && git commit -m "..." && git push`. |
| Kaggle account with phone verification | Needed for GPU and internet in notebooks. |
| DistilBERT trained on Kaggle (3 seeds) | D34, D35. Notebook: `notebooks/kaggle_distilbert.ipynb`. |
| Data download and classical training | `prepare`, `train-classical --variants base negmark negmark_w3 unigram` and `all` have run (D26). Sizes: train 63,971 / held-out 3,378 / val 872 / test 1,820 / negation 519 / IMDb 10,000. |
| Proposal's facts corrected | The repo copy is [proposal.md](proposal.md); H2/H3's "single-word features" are tested exactly as worded (D42). |
| Random forest | Trained fine (about 4 min per fit); its files are large (822 MB per variant). |
| Behavioral word lists reviewed | "messy" → "sloppy", everything else kept (D33). |
| Sanity checks of the numbers | Checked by Claude (D35). |
| External models run on Kaggle | D46. Notebook: `notebooks/kaggle_external.ipynb`. |

### If you ever re-run a Kaggle notebook
- Import it fresh (*File → Import notebook*, from the GitHub link); an older imported copy runs old
  code. Cell 2 prints `code version:`; it should be the latest commit on GitHub.
- Settings: **GPU T4 x1** and **Internet = On**. Internet can switch off when a session restarts,
  and without it the run fails within the first minute.
- *Save Version → Save & Run All*: a saved version keeps running after you close the laptop.
- Harmless messages: "Debugger warning: frozen modules…" at the start, and "Token indices sequence
  length is longer than … (535 > 512)" (reviews are cut to 512 tokens, start + end kept, D23).
- When it finishes: Output → download the zip → unzip into the **repo root** (`C:\claude\Mirage`),
  then ask Claude to check it and re-run `evaluate`, `warn` and `figures`.
