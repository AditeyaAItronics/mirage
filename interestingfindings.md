# Interesting findings beyond the hypotheses

Testing H1–H6 produced thirteen further findings. None was a hypothesis in the proposal, but each
one explains or extends one: *why* a hypothesis held, where it stops holding, or what the models
actually learned. This file gives each in full (the finding, the numbers, the reason, the
connection to the hypotheses, the caveats, the sources), as a source to pick from for the report.
The hypotheses themselves are answered in [hypothesisanswers.md](hypothesisanswers.md).

**How to read it.**
- Every number comes from a file in `results/`, named under "Sources"; "DNN" is the entry in
  [majordecisions.md](majordecisions.md) that explains the analysis. Brackets are 95% bootstrap
  confidence intervals.
- **Status** says whether the analysis was planned before any results were seen, or designed
  after the first results ("exploratory"). Exploratory findings are real measurements, but they
  should be reported as exploratory: we chose to look there because of what we had already seen.
- Model names as in [hypothesisanswers.md](hypothesisanswers.md): LR, MNB (Multinomial NB), GNB
  (Gaussian NB), RF, DT, DistilBERT (ours), HF DistilBERT and RoBERTa-large (external, ready-made).

## Overview

| § | Finding | Connects to | Status |
|---|---|---|---|
| 1 | Two models with the same accuracy behave very differently | H1, H6 | exploratory (score defined after first results) |
| 2 | DistilBERT answers "negative" when it can't read the sentiment word | H1 | exploratory (control added after the first result) |
| 3 | An independently trained DistilBERT has the same shortcut | H1, H5 | planned (second reference model) |
| 4 | A 5× larger model shrinks shortcuts but doesn't remove them | H1, H2 | planned; its accuracy is possibly contaminated |
| 5 | Naive Bayes judges a film by who made it, including names never tested | H1, H5 | exploratory (follow-up of the name test) |
| 6 | Negation marking fixes templates but not real sentences | H3 | exploratory (error analysis) |
| 7 | Being unusual is not the same as being wrong | H4 | planned (sanity check of the distance score) |
| 8 | The IMDb drop is about content, not length: full reviews are *easier* than their edges | H4, RQ on shift | exploratory (robustness check) |
| 9 | A model that can't read can't be fooled | H1, H6 | exploratory (found while building the score) |
| 10 | DistilBERT shares the classical models' sentiment words, and an "IMAX" shortcut | H5 | exploratory (extension of H5) |
| 11 | Confidence is honest for some models and not others | H4 | planned (calibration), connection exploratory |
| 12 | RoBERTa-large is *better* on IMDb than on SST-2: a contamination warning | H1, H4 (caveat) | exploratory |
| 13 | Negation marking makes the decision tree unstable on names | H3 | exploratory |
| 14 | What the findings have in common | all | — |
| 15 | The sentence frame matters as much as the model; DistilBERT can't read "would never call it great" | H1, H2, H3 | found by the pre-registered replication (D53) |
| 16 | Distance in DistilBERT's own features finds its errors, but only the ones confidence finds | H4 | planned (F2, D53) |

**Replication (D53).** Ten pre-registered claims about these findings were re-tested on fresh
data (10,000 new IMDb reviews and a second template suite with new sentence frames): §1 (claim
10), §2 (5, 6), §3 (7), §4 (8), §5 (9), §8 (14, 15), §11 (17) and §12 (16). **All ten
replicated**; the numbers are in `results/replication.csv` and the claims in
[docs/confirmatory_claims.md](docs/confirmatory_claims.md). The one claim that failed is about H2
(§15 explains it).

---

## 1. Leaderboard twins: the same accuracy, very different behaviour

**Finding.** Logistic regression and Multinomial NB are indistinguishable on the SST-2 test set
(83.5% vs 83.4% accuracy), but Multinomial NB's **robustness score** is 19 points lower (0.554 vs
0.743). The robustness score (D40) is the mean, over four checks, of the share of cases a model
gets *fully* right: negation template pairs, real negated sentences, typo pairs and name-swap
pairs. The gap between accuracy and robustness is the **Mirage Gap**.

**Table 1. Accuracy vs behaviour (base variants), sorted by Mirage Gap.**

| model | SST-2 test accuracy | negation templates | real negation | typos | names | robustness | Mirage Gap |
|---|---|---|---|---|---|---|---|
| Multinomial NB | 0.834 | 0.354 | 0.761 | 0.486 | 0.614 | 0.554 | **0.280** |
| Gaussian NB | 0.591 | 0.038 | 0.522 | 0.502 | 0.516 | 0.395 | 0.196 |
| Decision tree | 0.653 | 0.374 | 0.607 | 0.452 | 0.624 | 0.514 | 0.139 |
| Logistic regression | 0.835 | 0.700 | 0.773 | 0.508 | 0.992 | 0.743 | 0.091 |
| Random forest | 0.754 | 0.694 | 0.726 | 0.520 | 0.834 | 0.694 | 0.060 |
| HF DistilBERT | 0.923 | 1.000 | 0.892 | 0.610 | 1.000 | 0.876 | 0.047 |
| DistilBERT (ours) | 0.922 | 1.000 | 0.900 | 0.606 | 1.000 | 0.876 | 0.046 |
| RoBERTa-large | 0.946† | 1.000 | 0.931† | 0.708 | 0.984 | 0.906 | 0.040 |

† RoBERTa-large's accuracy (and its real-negation score, also SST-2 sentences) is possibly
contaminated: it was very likely trained on SST-2 and IMDb (§12, C12). Its template checks are
clean.

The twins differ on two checks: names (MNB 61% vs LR 99%) and negation templates (35% vs 70%).
Every model scores lower on behaviour than on accuracy; the size of the gap is what differs.

**Negation marking narrows the gap**, mostly through the templates: LR's gap falls from 0.091 to
0.040 (robustness 0.743 → 0.807), Multinomial NB's from 0.280 to 0.202.

**Why.** The two models read the same features but learn from them differently. Naive Bayes
estimates each word's class ratio *on its own*: a surname that happened to appear mostly in
negative training phrases keeps its full negative vote, however irrelevant it is. Logistic
regression fits all the weights *together* with regularisation, so a word whose label
association is already explained by the sentiment words around it is pushed towards zero. The
same difference makes Naive Bayes count "bad" in "not bad" at full strength.

**Connection to the hypotheses.** This is H1's claim, that accuracy hides systematic failures,
in its sharpest form: two models that no leaderboard can separate. It also complements H6. H6
showed that the *choice of Naive Bayes distribution* decides accuracy (59.1% vs 83.4%); this
shows that the *choice between a generative and a discriminative model* decides behaviour even
when accuracy is identical.

**Caveats.**
- The robustness score was defined after the first results, so it is exploratory (D40, G6).
- **Its ranking doesn't depend on the equal weights.** Leaving out each check in turn: for the 8
  base models, 3 of 4 leave-outs keep the order identical, and leaving out real negation gives
  Spearman 0.976 (two neighbours swap; top and bottom unchanged). For the other variants, 11 of 12
  leave-outs keep the order identical.
- An earlier version counted "didn't flip" as a pass, which made weak models look robust (§9).
  The score now requires both sentences of a pair to be right.

*Sources:* `results/mirage_gap.csv`, `results/mirage_gap_sensitivity.csv`,
`results/shortcut_fingerprint.csv`, `figures/mirage_gap.png`, `figures/shortcut_fingerprint.png`; D40.

---

## 2. DistilBERT's default: an unreadable sentiment word means "negative"

**Finding.** One typo in a positive adjective flips DistilBERT to negative in 65% of pairs, and
it moves confidently: P(positive) drops by 0.64 on average [0.53, 0.74]. For example, "The movie
was brilliant." scores 0.999 positive, and "The movie was brililant." scores 0.010. The same kind
of typo in a negative adjective flips only 9% of pairs. A **control test** puts the typo in the
noun instead ("The moive was brilliant."): DistilBERT flips **0%** of those pairs.

**Table 2. Typo pairs: share of pairs whose verdict flips, and mean change in P(positive) [95% CI].
Adjective typos: 271 positive and 229 negative sentences; noun typos: 237 and 263.**

| model | adjective typo, positive sentence | adjective typo, negative sentence | noun typo, positive | noun typo, negative |
|---|---|---|---|---|
| DistilBERT (ours) | **65%, −0.64** [−0.74, −0.53] | 9%, +0.11 [0.06, 0.17] | **0%**, −0.00 | **0%**, +0.00 |
| HF DistilBERT | **59%, −0.59** [−0.69, −0.48] | 15%, +0.15 [0.11, 0.20] | 0% | 0% |
| RoBERTa-large | 18%, −0.18 [−0.27, −0.10] | **42%, +0.42** [0.30, 0.54] | 0% | 0% |
| Logistic regression | 42%, −0.37 | 49%, +0.36 | 2%, +0.04 | 6%, −0.07 |
| Multinomial NB | 50%, −0.42 | 25%, +0.28 | 19%, +0.11 | 2%, −0.03 |
| Random forest | 18%, −0.13 | 59%, +0.21 | 8%, +0.03 | 14%, −0.03 |
| Decision tree | 31%, −0.21 | 14%, +0.08 | 11%, +0.01 | 19%, −0.07 |
| Gaussian NB | 11%, −0.02 | 17%, +0.14 | 15%, −0.08 | 20%, −0.06 |

**Why.** The control rules out "any odd word means negative": DistilBERT ignores a garbled noun
completely. The shortcut is narrower: when the *sentiment word itself* can't be read, the model
falls back on a default, and the default is negative. That default comes from the training data
(D37):
- DistilBERT's vocabulary splits rare or unknown words into fragments ("word pieces").
- In SST-2's training phrases, **16.0% of the words in negative phrases are split into
  fragments, against 10.9% in positive phrases**.
- Phrases containing any fragmented word are **49.9% positive**, against **61.4%** for phrases
  without one.
- The fragments come from coinages like "unwatchable", sarcastic spellings, and pieces like
  "##ible" from "horrible" and "terrible". So in training, "fragmented" leans negative.
- A typo turns "brilliant" into `br ##ili ##lan ##t`, which looks like that negative-leaning
  vocabulary.

The classical models behave differently: they simply lose the unknown word. Logistic regression
then drifts towards the middle from both sides (−0.37 / +0.36), with no default direction. The
tree models and Naive Bayes fall in between, each with its own lean.

**Connection to the hypotheses.** H1 predicted that typos would expose hidden failures
*including in DistilBERT*. That half of H1 holds (DistilBERT's typo error is 32 points above its
normal error), and this finding explains *why*: not a general fragility, but a learned default
that only shows when the sentiment word is lost.

**Caveats.**
- The control test (`typo_noun`) was added after the first typo result (exploratory).
- The fragment statistics are a correlation in the training data, the most likely source of the
  default, not a causal test (we didn't retrain on rebalanced data).

*Sources:* `results/typo_direction.csv`, `results/h1_test.csv`; D35, D37.

---

## 3. The shortcut replicates in an independently trained model

**Finding.** Hugging Face's DistilBERT fine-tuned on SST-2 (`distilbert-base-uncased-finetuned-sst-2-english`:
the same architecture, trained by someone else, with other settings) matches ours closely
everywhere:

| | SST-2 test | real negation | IMDb | IMDb edges | robustness |
|---|---|---|---|---|---|
| DistilBERT (ours) | 0.922 | 0.900 | 0.896 | 0.788 | 0.876 |
| HF DistilBERT | 0.923 | 0.892 | 0.893 | 0.793 | 0.876 |

It also has the **same typo shortcut**: 59% flips on positive-word typos, 15% on negative ones,
0% on nouns (Table 2). On SST-2 validation, McNemar finds no significant difference between the
two models (17 vs 23 sentences that only one gets right, Holm p 0.46).

**Why.** Two separate training runs on the same data learned the same default, so the default is
a property of SST-2, not of one training run, which agrees with the data analysis in §2.

**Connection to the hypotheses.** It strengthens H1's typo result: the failure is not an accident
of our model. It also extends H5's conclusion (the classical models share their shortcut words,
so the shortcuts come from the data) to the transformers: models trained on the same data inherit
the same shortcuts.

**Within our own DistilBERT, nothing depends on the random seed.** Every per-item result was rerun
on seeds 13, 42 and 87: all 291 checks agree (171 estimates keep their sign, 120 tests keep their
significance; D44).

**Caveat.** HF DistilBERT was trained on the full SST-2 training phrases, which include our
held-out phrases (held-out accuracy 0.989 vs 0.923 on test, C14); its validation and test numbers
are clean.

*Sources:* `results/main_table.csv`, `results/typo_direction.csv`, `results/mcnemar.csv`,
`results/seed_robustness.csv`; D39, D44, D46.

---

## 4. Does scale cure shortcuts? It shrinks them

**Finding.** RoBERTa-large (SiEBERT: about five times larger than DistilBERT, fine-tuned on 15
sentiment datasets) has the smallest Mirage Gap of all eight models (accuracy 0.946, robustness
0.906) and passes every negation template pair. But two shortcuts remain:
- **Typos still break it, in the opposite direction.** It flips 42% of *negative* sentences with
  a typo, against 18% of positive ones (Table 2): its default for an unreadable sentiment word
  leans *positive*. Its typo error is still 24 points above its normal error (H1 table).
- **Names may still move it, slightly.** It flips 1.0% of name-swap pairs (mean shift +0.010
  [0.000, 0.029]). With item-level intervals this was significant (Holm p 0.02); with the
  cluster intervals of D52 it is not (Holm p 0.31), so on the name-swap test alone RoBERTa's
  name effect is not established. Across all 35 film people in the name probe (§5), its reaction to
  a name follows the name's reputation in SST-2: ρ = 0.46 [0.06, 0.79] on the 23 never-tested
  names, Holm p 0.03. But the largest effect is only ±0.03 in probability, against ±0.32 for
  Multinomial NB.

**Why.** More data and parameters let the model read negation in context and rely less on single
words. Its default for unreadable words most likely points the other way because its training
data (15 datasets, not SST-2 alone) pairs fragmented words with labels differently; we did not
test this directly.

**Connection to the hypotheses.** It supports H2 at a larger scale: the transformer advantage on
negation holds and grows (0 template errors; 6.9% errors on real negated sentences against
DistilBERT's 10.0%). It qualifies H1: hidden failures persist even in a much stronger model; they
get smaller and change direction rather than disappearing.

**Caveat.** SiEBERT was very likely trained on SST-2 and IMDb, so its *accuracy* may be inflated
(§12, C12). The behavioural tests use sentences we generated, so the shortcut findings are not
affected by that.

*Sources:* `results/mirage_gap.csv`, `results/typo_direction.csv`, `results/name_effect.csv`,
`results/name_probe.csv`, `results/h1_test.csv`; D39, D45, D46.

---

## 5. Naive Bayes judges a film by who made it

**Finding.** In the name-swap test, changing only a film person's name ("Another Binoche comedy,
and it was decent." → "Another Hopkins comedy, and it was decent.") flips Multinomial NB's verdict
in **38% of pairs** (mean shift +0.37 [0.24, 0.49], 500 pairs). The other models barely react:

| model | name pairs flipped | mean shift in P(positive) [95% CI] |
|---|---|---|
| Multinomial NB | **37.6%** | +0.370 [0.244, 0.489] |
| Logistic regression | 0.8% | +0.064 [0.056, 0.074] |
| Gaussian NB | 3.8% | +0.013 [0.007, 0.021] (Holm p 0.004, below the 2-point minimum) |
| Random forest | 2.4% | +0.008 |
| Decision tree | 1.2% | −0.001 (n.s.) |
| RoBERTa-large | 1.0% | +0.010 |
| DistilBERT (ours), HF DistilBERT | 0% | ≈ 0 |

**Does it generalise beyond the six tested name pairs? (D45).** We placed all 35 film people with
at least 15 SST-2 training phrases into the *same* 40 neutral sentences, so only the name differs.
A model that learned names' reputations should react to each name in proportion to how positive
that name's training phrases were (its training log-odds). The test is the rank correlation
between the two, on the **23 names never used in any test** (out of sample), with the effect size
next to it, because a correlation of near-zero effects can look large:

| model | largest name effect | ρ, 23 never-tested names [95% CI] | ρ, all 35 names [95% CI] |
|---|---|---|---|
| **Multinomial NB** | **±0.32** | **0.91** [0.78, 0.96], Holm p < 0.001 | 0.97 [0.92, 0.99] |
| Logistic regression | ±0.13 | 0.33 [−0.13, 0.71] | **0.45** [0.10, 0.71], Holm p 0.03 |
| RoBERTa-large | ±0.03 | 0.46 [0.06, 0.79], Holm p 0.03 | 0.30 [−0.05, 0.65] |
| Random forest | ±0.03 | 0.45 [−0.02, 0.76] | 0.42 [0.05, 0.68] |
| Gaussian NB, decision tree | ≤ ±0.03 | not significant | not significant |
| DistilBERT (ours) | ±0.0009 | noise | noise |
| HF DistilBERT | ±0.00003 | 0.51 on noise-level effects | — |

**Why.** A name carries no sentiment, but in SST-2 each director or actor appears in phrases with
a particular balance of labels. Naive Bayes turns that balance directly into a vote (§1), so it
has in effect memorised each person's reputation, for names it was never tested on as much as for
the tested ones. Logistic regression shrinks those weights but keeps a weaker version. A
contextual model learns that the name isn't what decides the label.

**Connection to the hypotheses.** It is a hidden failure of exactly the kind H1 describes:
accurate on the test set, for a reason that breaks on new films. It also illustrates H5's
mechanism: the shortcut words the models share are words whose training labels happen to be
skewed. And it shows why the project reports an effect size next to every correlation: HF
DistilBERT's ρ = 0.51 sits on effects below 0.0001, which is noise.

**Caveats.**
- The probe was designed after the first name result (exploratory), but its 23 test names were
  never used before, so it is a genuine out-of-sample check.
- 35 names is a small sample for a correlation; the CIs are correspondingly wide for the weaker
  models.

*Sources:* `results/name_effect.csv`, `results/name_probe.csv`, `results/name_probe_by_name.csv`,
`results/name_associations.csv`; D32, D45.

---

## 6. Why negation marking works on templates but not on real sentences

**Finding.** Marking the words after a negation (`not good` → `not NOT_good`, up to the next
punctuation) closes **96%** [93, 99] of logistic regression's remaining negation gap to DistilBERT
on the directional template, but only **23%** [−2, 43] (not significant) on the 519 real negated
SST-2 sentences. On the real sentences, LR's error falls only from 22.7% to 19.8% (+2.9 points
[−0.2, +5.8], Holm p 1.0). Marking **fixes 38** of LR's 118 errors there, but **breaks 23**
sentences it had right.

**Why.** An error analysis of LR found three reasons (D38):
1. **Scope swallows whole clauses.** A real negation marks **6.9 words** on average, against 2.1
   in a template. In "it 's never a good sign when a film 's star spends the entirety of the
   film…", everything after "never" becomes `NOT_a`, `NOT_film`, `NOT_the`, and the signal on
   `NOT_good` is diluted.
2. **Real negation often doesn't negate the sentiment.**
   - Double negation: "it 's almost impossible not to be moved", which is positive.
   - Idioms: "it wo n't be long before you 'll spy it at a video store".
   - Negated non-sentiment verbs: "surely can not last".
   Marking pushes all of these the wrong way.
3. **Rare marked words have noisy weights.** `NOT_good` (−3.43) and `NOT_funny` (−4.29) are
   sensible, but `NOT_brilliant` is +1.07 (rare, so noisy) and `NOT_boring` is −0.08 (no signal).

A first explanation, that real sentences mark words the model never saw marked, was tested and
**rejected**: the model knows about as many of the marked words in real sentences (78%) as in
templates (74%).

**A second scope rule** (`negmark_w3`: at most 3 words, skipping fixed phrases such as "not only"
and "no doubt") gives the same picture: large template gains, small real-sentence gains (+2.9
[+0.8, +5.2] for LR, not significant after Holm). Its one significant side effect: LR's SST-2 test
accuracy improves by 1.4 points (D31).

**Connection to the hypotheses.** This explains why **H3 is only partly supported**. Templates
test negation in its easiest form, one negation word right before one sentiment word, which is
exactly what a token-level rule captures. Real negation is long-range, idiomatic and sometimes
double, so word pairs and marking together close 40% of LR's real-sentence gap, not all of it. A
contextual model handles that (0% template errors, 10% real-sentence errors), which is H2's result
seen from the other side.

*Sources:* `results/negation_gap.csv`, `results/negmark_effect.csv`, LR predictions on `sst2_neg`;
D31, D38, D42.

---

## 7. Being unusual is not the same as being wrong

**Finding.** The embedding distance from the SST-2 training data separates IMDb reviews from SST-2
test sentences well (AUROC **0.926**), so it detects the dataset shift. But *within* IMDb it does
not pick out the reviews a model gets wrong: AUROC **0.49–0.53** for all eight models, chance
level. Low confidence picks them out for every model (**0.55–0.86**). Review length, as a baseline,
does as badly as distance (0.49–0.54).

**Why.** Distance measures how unlike the training text an input is: its topic, length and style.
Almost every IMDb review is unlike SST-2's short phrases, so the distance is high for nearly all
of them and says little about which ones are hard. Errors most likely come from the content (mixed
sentiment, sarcasm, a twist at the end), which the model's own confidence partly reflects and an
external embedding doesn't see. The embedding model is also not the classifier: what is
unusual to MiniLM need not be hard for DistilBERT. Measured in DistilBERT's own features instead,
unusual *does* go with wrong (AUROC 0.82), but only for the reviews its confidence already flags
(§16).

**Connection to the hypotheses.** This is the reason behind **H4's rejection**. The distance works
as a shift detector, so the negative result isn't a broken score; it is the wrong tool for
spotting individual errors. The practical lesson for deployment: monitor distance to know *that*
the data has changed, and confidence to know *which* predictions to check.

The live demo shows this honestly: its "unusual input" flag says "unusual doesn't mean wrong", and
only the low-confidence flag is presented as an error warning (D41).

*Sources:* `results/early_warning.csv` (`auroc`, `ood_sanity_auroc`), `results/early_warning_sanity.csv`;
D25, D30.

---

## 8. The IMDb drop is about content, not length

**Finding.** Every model is worse on IMDb than on SST-2. To see why, we built a second IMDb set:
each review's **first and last sentence** only (`imdb_edges`, 44 words on average, close to SST-2's
19, with the review's label). Comparing the three sets splits the drop into two parts (D31):
- **length and format** = error on full reviews − error on their edges (same reviews, paired);
- **content** = error on the edges − error on SST-2 test (different sentences).

**Table 8. Error rates and the two parts of the shift, in points [95% CI]. All significant after Holm.**

| model | SST-2 test | IMDb edges | IMDb full | length / format | content |
|---|---|---|---|---|---|
| DistilBERT (ours) | 7.8% | 20.7% | 10.0% | **−10.7** [−11.5, −9.8] | **+12.9** [11.3, 14.3] |
| HF DistilBERT | 7.7% | 20.7% | 10.7% | −10.0 | +13.0 |
| RoBERTa-large† | 5.4% | 12.9% | 4.4% | −8.4 | +7.4 |
| Logistic regression | 16.5% | 28.1% | 20.5% | −7.7 [−8.7, −6.6] | +11.6 [9.6, 13.4] |
| Multinomial NB | 16.6% | 29.8% | 24.0% | −5.8 | +13.1 |
| Random forest | 24.6% | 33.4% | 29.7% | −3.7 | +8.8 |
| Decision tree | 34.7% | 41.7% | 38.8% | −2.9 | +7.0 |
| Gaussian NB | 40.9% | 48.1% | 45.4% | −2.6 | +7.1 |

† Possibly contaminated (§12): RoBERTa-large was very likely trained on SST-2 and IMDb.

**What it means.** The length part is *negative* for every model: a full review is *easier* than
its first and last sentences, because it contains more evidence. The whole drop comes from the
content, and part of the length advantage hides it: the full-review drop (e.g. DistilBERT +2.2
points) is the sum of a large content cost (+12.9) and a large length benefit (−10.7). Short-phrase
training did not make the models bad at long text.

**Connection to the hypotheses.** It sharpens the setting of H4: the shift that the distance
detects is mostly length and style, which does *not* cause errors, while the errors come from
content, which the distance doesn't see (§7). It also answers the proposal's question about how
the models cope with the SST-2 → IMDb shift.

**Caveat.** The edges keep the whole review's label, but two sentences don't always carry it
(B10), so part of the "content" cost is label noise in the edges set. The direction of the result
(length helps) is not affected by this.

*Sources:* `results/shift_decomposition.csv`, `results/main_table.csv`, `results/imdb_by_length.csv`;
D31, D35, D46.

---

## 9. A model that can't read can't be fooled

**Finding.** The two weakest models make *fewer* errors on typo tests than on normal sentences:
the decision tree 10.8 points fewer, Gaussian NB 25.4 points fewer (H1 table). Gaussian NB also
passed 86% of the typo pairs under an early version of the robustness score, and the decision
tree came out "more robust" (79%) than it was accurate (65%).

**Why.** A typo test asks "does the model change its answer when the sentiment word is
misspelled?". A model that barely uses the sentiment word in the first place doesn't change its
answer, so it passes, for the wrong reason. Gaussian NB's word effects are spread over 300 SVD
components (H5, H6), and the decision tree splits on only a few words, so losing one word often
changes nothing for them.

**What we changed because of it.** The robustness score (§1) counts a pair as passed only if the
model gets *both* sentences right, so robustness can't exceed competence. The decision tree then
drops to 37% on negation and 45% on typos, and every model sits below the accuracy diagonal.

**Connection to the hypotheses.** It qualifies H1's test: an error rate *lower* than normal on a
perturbation test is not evidence of robustness when the model is weak. It is also another face of
H6: Gaussian NB is so poor at reading individual words that it can't even be fooled by them.

*Sources:* `results/h1_test.csv`, `results/errors_by_type.csv`, `results/mirage_gap.csv`; D40, D47.

---

## 10. DistilBERT shares the classical models' sentiment words

**Finding.** H5 tested the classical models. Measured the same way (remove a word, measure the
change in P(positive), corrected for each model's reliability), **DistilBERT agrees with LR, random
forest and Multinomial NB at 0.71–0.77**: not as closely as they agree with each other
(0.88–0.92), but far more than "different words" (D43).

One shared word is a pure shortcut: **"IMAX"**. It appears in 54 SST-2 training phrases, 96% of
them positive (the training average is 56%). Removing "IMAX" from a test sentence drops
DistilBERT's P(positive) by **0.31** on average, its 10th most influential word of all, and LR's by
0.21 (its 13th). A film's screen format says nothing about whether the film is good.

**Why.** The models learned from the same labels, and SST-2's labels make some words reliably
positive or negative, sentiment words and incidental ones like "IMAX" alike. DistilBERT differs
from the classical models where context matters (negation, H2), not in which words carry sentiment.

**Connection to the hypotheses.** It extends H5 beyond the classical models: the shortcut words
come from the data, and even a contextual model picks some of them up. It corrects an earlier
reading (D35, from D16's top-50 comparison) that DistilBERT used "almost entirely different
words"; that came from comparing different explanation methods, not different models.

**Caveats.** "IMAX" occurs in only 5 test sentences, so its effect is an illustration rather than a
measured estimate; DistilBERT's own word effects are only moderately reproducible (split-half
reliability 0.52), which the correction accounts for.

*Sources:* `results/explain/occlusion_agreement.csv`, `results/explain/occlusion_reliability.csv`,
`results/explain/occlusion_importance.csv`, `results/explain/overlap_occlusion.csv`; D43.

---

## 11. Confidence is honest for some models and not others

**Finding.** Calibration (whether a model's 80% means right 80% of the time; expected calibration
error, ECE, lower is better) and error warning (whether low confidence picks out errors; AUROC) are
different properties:

| model | ECE, SST-2 test | ECE, IMDb | error-warning AUROC, IMDb |
|---|---|---|---|
| RoBERTa-large† | 0.052 | 0.043 | 0.857 |
| DistilBERT (ours) | 0.046 | 0.051 | 0.836 |
| Random forest | 0.068 | **0.057** | 0.702 |
| Logistic regression | **0.016** | 0.081 | 0.751 |
| HF DistilBERT | 0.064 | 0.082 | 0.845 |
| Decision tree | 0.144 | 0.200 | 0.562 |
| Multinomial NB | 0.122 | **0.229** | 0.688 |
| Gaussian NB | 0.365 | 0.376 | 0.547 |

- **Random forest is the best-calibrated classical model on IMDb** (0.057), better than LR (0.081)
  despite being 9 points less accurate: averaging 300 trees gives naturally moderate probabilities.
- **LR is the best-calibrated model of all in-domain** (0.016) but loses that on IMDb.
- **Multinomial NB is badly overconfident** (0.229 on IMDb): multiplying thousands of word
  probabilities pushes its outputs to 0 or 1 (D26). Yet its low confidence still flags errors
  (0.688): its probabilities are badly *scaled* but still *ranked* sensibly.

**Connection to the hypotheses.** It supports the practical side of H4's answer: confidence is a
usable warning signal even for a poorly calibrated model, because error detection only needs the
ranking. For a threshold ("check every prediction below 70%"), calibration matters too, and there
RF and the transformers are the safe choices.

**Caveat.** ECE depends on the binning (G3).

*Sources:* `results/main_table.csv` (`ece__*`), `results/shortcut_fingerprint.csv`,
`results/early_warning.csv`; D26, D40.

---

## 12. RoBERTa-large is better on IMDb than on SST-2: a contamination warning

**Finding.** Every model drops on IMDb (DistilBERT −2.6 points, LR −3.9, MNB −7.3), except
RoBERTa-large, which *improves*: 94.6% on SST-2 test, **95.6%** on IMDb.

**Why it matters.** IMDb reviews are a harder, shifted domain for every model trained on SST-2
(§8). A model that does better there was most likely trained on IMDb (SiEBERT's training data
includes many review datasets). So its IMDb accuracy, and possibly its SST-2 accuracy, overstate
how it would do on unseen data (C12).

**Connection to the hypotheses.** It is a caveat for every result that uses RoBERTa-large's
accuracy (H1's comparison sets, H4's IMDb errors, the Mirage Gap), not for its behavioural tests,
which use sentences we generated. It is also a small example of the project's theme: a high score
on a benchmark doesn't tell you what the model has seen.

*Sources:* `results/main_table.csv` (`accuracy__*`, `imdb_acc_drop`); C12, C14, D46.

---

## 13. Negation marking makes the decision tree unstable on names

**Finding.** With negation marking, the decision tree flips on **24.2%** of name-swap pairs (1.2%
without it, 4.0% with the 3-word scope), yet its *mean* shift stays at zero (−0.002 [−0.017,
0.016]). The flips go both ways and cancel out: no name bias, but an unstable model.

**Why (likely, not tested directly).** Marking adds many new rare features (`NOT_…`) to the
vocabulary. A single tree's few splits are sensitive to which features it picks; with marking, its
path through the tree changes for small reasons, such as which name is in the sentence.

**Connection to the hypotheses.** It is a side effect relevant to H3: a feature that helps
negation can make another behaviour worse. It also shows why the project reports both a flip rate
(how often the answer changes) and a mean shift (whether it changes in a consistent direction):
either alone would have told half the story.

*Sources:* `results/name_effect.csv`; D29, D32.

---

## 14. What the findings have in common

Three points connect them:
- **Shortcuts come from the training data.** Names' reputations (§5), fragmented words (§2),
  "IMAX" (§10) and their replication across training runs (§3) all trace back to label imbalances
  in SST-2.
- **The model family decides how strongly a shortcut is used.** Naive Bayes uses every skewed word
  at full strength; logistic regression shrinks them; transformers mostly read context but keep
  defaults for what they can't read (§1, §2, §4).
- **The tests themselves need testing.** Re-running the claims on fresh data and new sentence
  frames confirmed 16 of 17 and caught one that leaned on the frames (§15): a template result is
  only as general as its sentences.
- **Accuracy hides all of it.** The leaderboard twins (§1), the larger model's remaining shortcuts
  (§4) and the contamination warning (§12) are invisible without behavioural tests, which is the
  project's central claim.

---

## 15. The sentence frame matters as much as the model

**Finding.** The behavioural tests put the same adjectives into several sentence frames (3–4
per test). The second suite (D10, frame set B) uses four new frames, and splitting every
failure rate by frame (`results/behavioral_by_frame.csv`) shows how much the frame alone moves
it. Logistic regression's directional-negation failure rate (single words + pairs):

| frame | sentence pair | LR fails |
|---|---|---|
| A2 | I thought the movie was excellent. → I don't think the movie was excellent. | 0% |
| A1 | This movie is good. → This movie isn't good. | 6% |
| B2 | Overall, the movie was great. → Overall, the movie wasn't great. | 26% |
| A0 | The movie was wonderful. → The movie was not wonderful. | 55% |
| B3 | I would call the movie great. → I would never call the movie great. | 63% |
| B1 | The movie felt great. → The movie didn't feel great. | **100%** |

The frames differ significantly (chi-square, Holm) in 29 of the 72 model × test × suite cells
that can be tested, mostly in the negation tests and mostly for the bag-of-words models.

**The "Sadly," check.** One new negation frame starts with "Sadly," on purpose (it is mildly
negative itself). It helps the bag-of-words models exactly as feared: on "Sadly, the movie was not
great." LR fails 0% of cases against 27–31% on the plain frames, Multinomial NB 21% against
40–44%, Gaussian NB 60% against 84–100%. Their negation scores on that frame are partly the word
"sadly", not the negation.

**DistilBERT's blind spot.** Both DistilBERTs read every frame right except one: "I would never
call the movie great." is positive for our DistilBERT in **96%** of cases [90, 100] and for HF
DistilBERT in 99%. RoBERTa-large fails 7% [0, 17]. Every other new negation ("I did not find
the movie good", "The movie wasn't great", "I can't say the movie was good") gets 0 errors from
our DistilBERT.

**Why.** A bag-of-words model's verdict on a negated sentence depends on which word pairs it
has seen in training: "isn't good" and "don't think" are common SST-2 pairs with a clear
label; "didn't feel" is not, so LR falls back on "great". DistilBERT handles negation in
context, but "would never call" puts the negator two words before the verb and four before the
adjective, a construction that is probably rare in SST-2's short phrases; RoBERTa-large,
trained on 15 datasets, has seen enough of it. We didn't test the training-data explanation
directly.

**Connection to the hypotheses.**
- **H2:** this is why the pre-registered claim 3 failed: with DistilBERT failing B3, the decision
  tree's and random forest's gaps fall below 10 points (details in
  [hypothesisanswers.md, H2](hypothesisanswers.md#h2--distilbert-handles-negation-much-better)).
- **H1, H3:** every template number is an average over a few frames, and the frames differ by up
  to 100 points. The verdicts that held on both suites are the robust ones; the size of a
  template effect should be read as frame-dependent.
- It supports §4: scale helps exactly where the smaller transformers fail.

**Caveat.** Frame B3 is one frame with 119 pairs; it shows a construction DistilBERT fails, not
that *never* in general is a problem. The 28 name-swap sentences are the same in both suites
(name-swap frames weren't paraphrased); no claim uses them.

*Sources:* `results/behavioral_by_frame.csv`, `results/replication.csv`,
`results/predictions/*__behavioral_v2.parquet`; D10, D53.

---

## 16. Distance in DistilBERT's own features finds its errors, but only the ones confidence finds

**Finding.** Measured in DistilBERT's own [CLS] features instead of an outside embedding (F2,
Lee et al.), the distance from SST-2 picks out DistilBERT's IMDb errors with AUROC **0.822**
[0.809, 0.837], against 0.53 with MiniLM and 0.836 for confidence. Combined with confidence it
adds **+0.000**. For the classical models, distance in their own TF-IDF features stays at chance
(0.47–0.53); only Multinomial NB gains from the combination (+0.034), as in H4.

**Why.** In its own features, a review DistilBERT gets wrong does look unusual, but the same
features produce its confidence, so "unusual" and "unsure" are the same signal measured twice.
An outside embedding sees a different kind of "unusual" (topic and style), which has nothing to
do with the errors.

**Connection to the hypotheses.** It makes the H4 rejection stronger: the result is not just
that our distance was a bad error detector. Even a distance that detects errors well adds
nothing beyond confidence. Details and the full table are in
[hypothesisanswers.md, H4](hypothesisanswers.md#h4--does-embedding-distance-predict-errors-beyond-confidence).

*Sources:* `results/early_warning_own.csv`; D53.
