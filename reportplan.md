# Report plan

How the final report gets written: what only you can do, what Claude does, and the skeleton of the
report with the source of every section. The proposal promises a **6–8 page report answering the
four research questions with tables, charts and 8–10 explained failure examples**
([proposal.md](proposal.md)).

Every number in the report comes from a file in `results/` and every decision is explained in
[majordecisions.md](majordecisions.md) (D-numbers below), so the report can be checked line by line.

---

## Part 1: the walkthrough (what happens, in order)

| Step | Who | What | Time |
|---|---|---|---|
| 1 | **You** | **Tell Claude the format rules:** page limit, required template (Word / LaTeX / course template), citation style, required sections (e.g. ethics statement, contribution statement), due date, and whether appendices count toward the limit. | 5 min |
| 2 | **You** | **Approve the hypothesis verdicts** (Part 3 below). How strongly to word each verdict is your academic call. | 15 min |
| 3 | Claude | Shortlist about 20 failure-example candidates in `report/failure_candidates.md`. For each: the sentence, what each model said, and which shortcut it shows. | — |
| 4 | **You** | **Pick 8–10 of them** and approve (or rewrite) the one-line explanation of each. Choosing the examples that tell the story is your judgment, and you'll be asked about them. | 30 min |
| 5 | **You** | **Decide main text vs appendix** for the extras: the external models (D46), the Mirage Gap (D40), the name probe (D45), the live demo (D41). Recommendation in Part 2. | 5 min |
| 6 | Claude | Draft the report section by section in the agreed format, with every figure and table, and captions that state the result. | — |
| 7 | Claude | **Number check:** a script compares every number in the draft with `results/*.csv` and flags any mismatch. | — |
| 8 | **You** | **Review the draft.** Read it as the examiner would; mark anything unclear, overstated or not in your voice. Claude revises. One or two rounds. | 1–2 h |
| 9 | **You** | **Write or approve the personal parts:** abstract wording, acknowledgements, anything the course requires in your own words. | 30 min |
| 10 | Claude | Final pass: references, figure numbering, page count, export (Word / PDF). | — |
| 11 | **You** | Submit. | — |

**What Claude needs from you before starting (steps 1, 2 and 5):** a short message with the
format rules, "verdicts approved" (or your changes), and your main-text / appendix choice. Steps 3
and 6 can then run straight away.

**Presentation slides** are a separate deliverable. They can reuse the report's figures; ask when
the report draft is done.

---

## Part 2: the report skeleton

Target: 8 pages of main text plus appendices. Page budgets are approximate.

### 0. Title and abstract (≈150 words)
*Mirage: Exposing the Illusion of Accuracy. Shortcut learning in classical, Gaussian and
transformer sentiment classifiers.*
The abstract gives four sentences of context and method, then the headline results: the Mirage Gap,
the H1–H6 verdicts in one line each, and the "scale shrinks but doesn't remove shortcuts" finding.

### 1. Introduction (½ page)
- The problem: high accuracy can hide shortcut learning. Three motivating failures ("not good",
  a typo, a name).
- **Teaser figure: the Mirage Gap** (`figures/mirage_gap.png`, D40). Multinomial NB "looks 83%,
  behaves 55%".
- Contributions, as a bullet list: a behavioral suite of 3,750 cases; a leak-free evaluation; one
  word-ranking rule for every model family; a formal test of each hypothesis.

### 2. Background (½ page)
Shortcut learning (Geirhos; McCoy), CheckList (Ribeiro), negation tagging (Pang), max-probability
error detection (Hendrycks and Gimpel), Mahalanobis OOD (Lee), calibration (Guo). These come from
the proposal's reference list.

### 3. Data and models (¾ page)
- **Table 1, datasets:** SST-2 train / held-out / validation / test (public labels, D22), the
  negation subset (519), IMDb (10,000, capped, D23) and its first + last sentence version (D31),
  the behavioral suite (3,750 cases).
  - Sources: `results/data_stats.csv`, D21–D23, D31.
- **One paragraph on leakage:** the sentence-grouped held-out split (D21) and dropping test
  sentences seen in training. This answers the first question a careful examiner asks.
- **Table 2, models:** the five classical models and DistilBERT. The feature ladder (single words →
  + word pairs → + negation marking, D42) and the two external models (D39) as rows.
  - Sources: `results/tuning_classical.csv`, D29, D34–D36.

### 4. Method (1 page)
- **Behavioral tests:** each test type with one example sentence, and how it is scored (MFT, INV,
  DIR). Sources: D24, D27, D32, D37.
- **Statistics:** bootstrap CIs, paired vs independent comparisons, Holm within each family, and
  McNemar. Sources: D12, D28, D47.
- **Early warning:** MiniLM embeddings, Gaussian distance, scoring each review by its most unusual
  sentence, and length as a baseline. Sources: D25, D30.
- **Explanations:** model-specific explanations for illustration. For H5, one rule for every model
  (remove a word, measure the change) and reliability-corrected agreement. Sources: D16, D43.
- **The robustness score:** defined, and stated as defined after the first results. Source: D40.

### 5. Results, one subsection per hypothesis (3 pages)

| § | Hypothesis | Verdict | Main figure / table | Source |
|---|---|---|---|---|
| 5.1 | **H1** hidden failures | partly supported | Table: error differences vs negation-free test sentences; heatmap `error_heatmap_base.png` | `h1_test.csv`, D47 |
| 5.2 | **H2** negation gap | supported | the ladder figure `negation_gap_sst2_neg.png` | `negation_gap.csv`, D42 |
| 5.3 | **H3** pairs + marking close part of it | supported for LR (40%, CI 21–58%) | same figure + a short table | `negation_gap.csv`, D38, D42 |
| 5.4 | **H4** distance vs confidence | rejected | AUROC table; `error_vs_score_distilbert.png` | `early_warning.csv`, D30 |
| 5.5 | **H5** shared shortcut words | supported for 3 of 5 | `h5_word_effect_agreement.png` | `occlusion_agreement.csv`, D43 |
| 5.6 | **H6** Gaussian vs Multinomial NB | supported | two lines of numbers + McNemar | `main_table.csv`, `mcnemar.csv` |

Each subsection follows the same pattern: the claim, the test, the number with its CI, and one
sentence on what it means.

### 6. Beyond the hypotheses (1 page)
**Source material, all numbers checked against `results/*.csv`:**
- [hypothesisanswers.md](hypothesisanswers.md): H1–H6 in full (claim, test, tables with CIs,
  verdict, why, caveats, robustness checks), for §5.
- [interestingfindings.md](interestingfindings.md): 13 findings beyond the hypotheses, each tied
  to the hypothesis it explains or extends, with its status (planned / exploratory), for §6 and
  the Discussion.

Both are deliberately longer than the report: pick what goes in, and how much.
The **Shortcut fingerprint** figure (D40, `shortcut_fingerprint.png`) fits next to 6.1.
- **8–10 explained failure examples** (from step 4), as a table: sentence, model, prediction, the
  shortcut.

**Recommendation for step 5:** keep the Mirage Gap and the external models in the main text
(they're the presentation's story), put the name probe's detail and the demo in an appendix.

### 7. Discussion (¾ page)
- **RQ2, "when is a cheap model good enough?":** logistic regression matches the transformers on
  in-domain accuracy minus 9 points, behaves well on names, but fails negation. Choose by the
  failure that matters for the application.
- **RQ3:** confidence is a usable warning signal and embedding distance isn't. Practical advice
  for deployment.
- **Why negation marking fails on real sentences** (D38): long-range, idiomatic, sometimes double.
- **What the shortcuts have in common:** they come from the training data, since H5 holds and
  HF DistilBERT shares our typo shortcut.

### 8. Limitations (½ page)
From [shortfalls.md](shortfalls.md), stated plainly:
- RoBERTa-large's possible contamination (C12).
- English movie reviews only (A3).
- Template tests are hand-written (D1, mitigated).
- Removing single words misses interactions (E4).
- IMDb excerpt labels are noisy (B10).
- DistilBERT isn't tuned and has 3 seeds (C3, C4).
- Sarcasm is out of scope (D4).

The analysis choices made after the first results (the robustness score D40, the typo-noun
control D37, the name probe D45) are labelled **exploratory**.

### 9. Conclusion and future work (¼ page)
Three sentences of conclusion. Future work comes from the proposal: toxicity / identity terms,
newer models, counterfactual augmentation, multilingual shift.

### Appendices
- **A. Robustness checks:**
  - every result on all three DistilBERT seeds (D44, `seed_robustness.csv`);
  - the robustness score's leave-one-out (D40);
  - the second negation-scope rule (D31);
  - the threshold check for H5 (D43).
- **B. The behavioral suite:** every test type, its counts, examples, and the word lists (D33).
- **C. Full tables:** the main table for all models and variants, errors by type, and McNemar.
- **D. The name probe in detail** (D45) and **the live demo** (D41).
- **E. Reproducibility:** commands, the Kaggle notebooks, `requirements-lock.txt` and the decision
  log.

---

## Part 3: verdicts to approve (step 2)

| | Proposed wording |
|---|---|
| **H1** | *Partly supported.* Typos clearly raise the error of every model that reads sentiment words, transformers included (DistilBERT +32 points). Negation clearly raises it for the bag-of-words models but not for the transformers. |
| **H2** | *Supported.* With single-word features, every classical model is significantly worse than DistilBERT on negation: by 14–35 points on real sentences, and on every template test. |
| **H3** | *Supported for logistic regression.* Word pairs and negation marking together close 40% (95% CI 21–58%) of its real-sentence gap: part, not all. On templates they close 25–98% for every model; on real sentences the other models barely benefit. |
| **H4** | *Rejected.* Embedding distance does not predict errors on IMDb (AUROC ≈ 0.5 for all eight models), while confidence does (0.55–0.86), even though distance clearly separates the two datasets. |
| **H5** | *Supported for the three main classical models.* Logistic regression, random forest and Multinomial NB agree on what each word does (0.88–0.92, reliability-corrected); Gaussian NB and the decision tree do not. |
| **H6** | *Supported.* Gaussian NB: 59.1% accuracy vs 83.4% for Multinomial NB (F1 0.589 vs 0.833; McNemar p < 10⁻⁵⁰). |
