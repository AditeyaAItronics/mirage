# Report plan

The skeleton of the final report: its sections, what goes in each, and the source of every
number. The proposal promises a **6–8 page report answering the four research questions with
tables, charts and 8–10 explained failure examples** ([proposal.md](proposal.md)).

Where everything else lives:
- **The steps to write it, yours and Claude's, in order:** [whatyouneedtodo.md](whatyouneedtodo.md),
  "The report".
- **The results, with every number and CI:** [hypothesisanswers.md](hypothesisanswers.md) (H1–H6,
  including the proposed verdict for each) and [interestingfindings.md](interestingfindings.md)
  (13 findings beyond the hypotheses). This file only points at them, so the numbers are kept in
  one place.
- **Why each analysis was done the way it was:** [majordecisions.md](majordecisions.md) (the
  D-numbers below). **Limitations:** [shortfalls.md](shortfalls.md).

---

## The skeleton

Page budgets are a suggestion for a 6–8 page main text; how much goes in is your call.

### 0. Title and abstract (≈150 words)
*Mirage: Exposing the Illusion of Accuracy. Shortcut learning in classical, Gaussian and
transformer sentiment classifiers.*
Four sentences of context and method, then the headline results: the Mirage Gap
([findings §1](interestingfindings.md#1-leaderboard-twins-the-same-accuracy-very-different-behaviour)),
the H1–H6 verdicts in one line each ([summary](hypothesisanswers.md#summary)), and "scale shrinks
shortcuts but doesn't remove them" ([findings §4](interestingfindings.md#4-does-scale-cure-shortcuts-it-shrinks-them)).

### 1. Introduction (½ page)
- The problem: high accuracy can hide shortcut learning. Three motivating failures ("not good",
  a typo, a name).
- **Teaser figure: the Mirage Gap** (`figures/mirage_gap.png`, D40): the leaderboard twins,
  logistic regression and Multinomial NB (findings §1).
- Contributions, as a bullet list: a behavioral suite of 3,750 cases; a leak-free evaluation; one
  word-ranking rule for every model family; a formal test of each hypothesis.

### 2. Background (½ page)
Shortcut learning (Geirhos; McCoy), CheckList (Ribeiro), negation tagging (Pang), max-probability
error detection (Hendrycks and Gimpel), Mahalanobis OOD (Lee), calibration (Guo). These come from
the proposal's reference list.

### 3. Data and models (¾ page)
- **Table 1, datasets:** SST-2 train / held-out / validation / test (public labels, D22), the
  negation subset (519), IMDb (10,000, capped, D23) and its first + last sentence version (D31),
  the behavioral suite (3,750 cases). Source: `results/data_stats.csv`; D21–D23, D31.
- **One paragraph on leakage:** the sentence-grouped held-out split (D21) and dropping test
  sentences seen in training. This answers the first question a careful examiner asks.
- **Table 2, models:** the five classical models and DistilBERT, the feature ladder (single words →
  + word pairs → + negation marking, D42) and the two external models (D39). Sources:
  `results/tuning_classical.csv`, `results/main_table.csv`; D29, D34–D36.

### 4. Method (1 page)
- **Behavioral tests:** each test type with one example sentence, and how it is scored (MFT, INV,
  DIR). Sources: D24, D27, D32, D37.
- **Statistics:** bootstrap CIs that resample clusters (adjectives, films), paired vs independent
  comparisons, Holm within each family, minimum effects that matter, and McNemar. Sources: D12,
  D28, D47, D52.
- **Replication:** 17 claims written down before scoring fresh data (10,000 new IMDb reviews, a
  second template suite with new frames). Source: D53, `docs/confirmatory_claims.md`.
- **Early warning:** MiniLM embeddings, Gaussian distance, scoring each review by its most unusual
  sentence, and length as a baseline; distance in each classifier's own features as a check
  (F2). Sources: D25, D30, D53.
- **Explanations:** model-specific explanations for illustration. For H5, one rule for every model
  (remove a word, measure the change) and reliability-corrected agreement. Sources: D16, D43.
- **The robustness score:** defined, and stated as defined after the first results. Source: D40.

### 5. Results, one subsection per hypothesis (3 pages)
Each subsection follows the same pattern: the claim, the test, the number with its CI, and one
sentence on what it means. Everything needed is in the linked section of
[hypothesisanswers.md](hypothesisanswers.md).

| § | Hypothesis | Answered in | Main figure / table | Source |
|---|---|---|---|---|
| 5.1 | **H1** hidden failures | [H1](hypothesisanswers.md#h1--hidden-failures-on-negation-and-typos) | Table H1; heatmap `error_heatmap_base.png` | `h1_test.csv`, D47 |
| 5.2 | **H2** negation gap | [H2](hypothesisanswers.md#h2--distilbert-handles-negation-much-better) | the ladder figure `negation_gap_sst2_neg.png` | `negation_gap.csv`, D42; claim 3 did not replicate (D53) |
| 5.3 | **H3** pairs + marking close part of it | [H3](hypothesisanswers.md#h3--word-pairs-and-negation-marking-close-part-of-the-gap) | same figure + Table H3a | `negation_gap.csv`, D38, D42 |
| 5.4 | **H4** distance vs confidence | [H4](hypothesisanswers.md#h4--does-embedding-distance-predict-errors-beyond-confidence) | Table H4; `error_vs_score_distilbert.png` | `early_warning.csv`, D30 |
| 5.5 | **H5** shared shortcut words | [H5](hypothesisanswers.md#h5--do-the-classical-models-share-their-shortcut-words) | `h5_word_effect_agreement.png` | `occlusion_agreement.csv`, D43 |
| 5.6 | **H6** Gaussian vs Multinomial NB | [H6](hypothesisanswers.md#h6--gaussian-nb-vs-multinomial-nb) | Table H6 | `main_table.csv`, `mcnemar.csv` |

### 6. Beyond the hypotheses (1 page)
Pick from the 13 findings in [interestingfindings.md](interestingfindings.md); its overview table
says which hypothesis each one explains and whether it was planned or exploratory. Suggested core:
- §1 the leaderboard twins and the Mirage Gap, with `shortcut_fingerprint.png` (D40);
- §2–§4 the typo default, its replication, and scale (D37, D46);
- §5 the name shortcut on never-tested names (D45);
- §15 the sentence frame matters, and DistilBERT's "would never call it great" blind spot, which is
  why claim 3 failed (D53);
- the replication table: 16 of 17 claims replicated (`results/replication.csv`);
- **8–10 explained failure examples** (chosen by you, see whatyouneedtodo.md), as a table:
  sentence, model, prediction, the shortcut.

**Suggestion for the main-text / appendix choice:** keep the Mirage Gap and the external models in
the main text (they're the presentation's story), put the name probe's detail and the demo in an
appendix.

### 7. Discussion (¾ page)
- **RQ2, "when is a cheap model good enough?":** logistic regression is 9 points behind the
  transformers in-domain, behaves well on names, but fails negation (findings §1, H2). Choose by
  the failure that matters for the application.
- **RQ3:** confidence is a usable warning signal and embedding distance isn't (H4, findings §7,
  §11). Practical advice for deployment.
- **Why negation marking fails on real sentences** (findings §6, D38).
- **What the shortcuts have in common** (findings §14): they come from the training data.

### 8. Limitations (½ page)
From [shortfalls.md](shortfalls.md), stated plainly:
- RoBERTa-large's possible contamination (C12; findings §12).
- English movie reviews only (A3).
- Template tests are hand-written (D1, mitigated).
- Removing single words misses interactions (E4).
- IMDb excerpt labels are noisy (B10).
- DistilBERT isn't tuned and has 3 seeds (C3, C4).
- Gaussian NB's comparison includes its SVD compression (C6).
- Sarcasm is out of scope (D4).
- Template effects depend on the sentence frame (D12); one pre-registered claim failed on new
  frames (G11).
- The minimum effects were set after the results (G9, D52).

The analysis choices made after the first results are labelled **exploratory**; the Status column
of interestingfindings.md's overview lists them.

### 9. Conclusion and future work (¼ page)
Three sentences of conclusion. Future work comes from the proposal: toxicity / identity terms,
newer models, counterfactual augmentation, multilingual shift.

### Appendices
- **A. Robustness checks:**
  - every result on all three DistilBERT seeds (D44, `seed_robustness.csv`);
  - the robustness score's leave-one-out (D40, `mirage_gap_sensitivity.csv`);
  - the second negation-scope rule (D31);
  - the threshold check for H5 (D43).
- **B. The behavioral suite:** every test type, its counts, examples, and the word lists (D33).
- **C. Full tables:** the main table for all models and variants, errors by type, and McNemar.
- **D. The name probe in detail** (D45) and **the live demo** (D41, D48, D49).
- **E. Reproducibility:** commands, the Kaggle notebooks, `requirements-lock.txt` and the decision
  log.
