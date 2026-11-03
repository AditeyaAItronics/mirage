# 6. Beyond the hypotheses: what else the tests revealed

<!-- Draft for report section 6 (reportplan.md, Part 2). Every number is taken from the result
file named in its source line; DNN = the entry in majordecisions.md that explains it.
Analyses designed after the first results are marked "exploratory", as the report plan requires. -->

Testing H1–H6 produced seven further findings. None of them was a hypothesis in the proposal, but
each one explains or extends one: *why* a hypothesis held, where it stops holding, or what the
model actually learned. Table 6.1 maps each finding to the hypothesis it connects to.

**Table 6.1. Findings beyond the hypotheses.**

| § | Finding | Connects to | Status |
|---|---|---|---|
| 6.1 | Two models with the same accuracy behave very differently | H1, H6 | exploratory (score defined after first results) |
| 6.2 | DistilBERT answers "negative" when it can't read the sentiment word | H1 | exploratory (control test added after the first result) |
| 6.3 | An independently trained DistilBERT has the same shortcut | H1, H5 | planned (second reference model) |
| 6.4 | A 5× larger model shrinks shortcuts but doesn't remove them | H1, H2 | planned; accuracy possibly contaminated |
| 6.5 | Naive Bayes judges a film by who made it, for names it was never tested on | H1, H5 | exploratory (follow-up of the name test) |
| 6.6 | Negation marking fixes templates but not real sentences | H3 | exploratory (error analysis) |
| 6.7 | Being unusual is not the same as being wrong | H4 | planned (sanity check of the distance score) |

---

## 6.1 Leaderboard twins: the same accuracy, very different behaviour

**Finding.** Logistic regression and Multinomial NB are indistinguishable on the SST-2 test set
(83.5% vs 83.4% accuracy), but Multinomial NB's robustness score is 19 points lower (0.554 vs
0.743). The difference comes from two of the four checks: it passes 61% of the name-swap pairs
against 99% for logistic regression, and 35% of the negation template pairs against 70%. Its
Mirage Gap (accuracy minus robustness) is the largest of all eight models, 28 points
(Figure `mirage_gap.png`).

**Why.** The two models read the same features but learn from them differently. Naive Bayes
estimates each word's class ratio *on its own*: a surname that happened to appear mostly in
negative training phrases keeps its full negative vote, however irrelevant it is. Logistic
regression fits all the weights *together* with regularisation, so a word whose label
association is already explained by the sentiment words around it is pushed towards zero. The
same mechanism makes Naive Bayes count "bad" in "not bad" at full strength.

**Connection to the hypotheses.** This is H1's claim, that accuracy hides systematic failures,
in its sharpest form: two models no leaderboard can separate. It also complements H6. H6 showed
that the *choice of Naive Bayes distribution* matters for accuracy (Gaussian 59.1% vs Multinomial
83.4%); this shows that the *choice between a generative and a discriminative model* matters for
behaviour even when accuracy is equal.

**Caveat.** The robustness score was defined after the first results (D40), so it is
exploratory. Its conclusion doesn't depend on the equal weighting: leaving out each of the four
checks in turn keeps the ranking of the models identical (Spearman 1.0 in all 12 checks).

*Sources:* `results/mirage_gap.csv`, `results/mirage_gap_sensitivity.csv`; D40.

## 6.2 DistilBERT's default: an unreadable sentiment word means "negative"

**Finding.** One typo in a positive adjective flips DistilBERT to negative in 65% of pairs, and
it moves confidently: the positive probability drops by 0.64 on average (95% CI 0.59–0.69), for
example "The movie was brilliant." 0.999 → "The movie was brililant." 0.010. The same typo in a
negative adjective flips only 9% of pairs. A **control test** puts the typo in the noun
instead ("The moive was brilliant."): DistilBERT flips **0%** of those pairs.

**Table 6.2. Typo pairs: share of pairs whose verdict flips, and mean change in P(positive).**

| model | typo in adjective, positive sentence | typo in adjective, negative sentence | typo in noun (control) |
|---|---|---|---|
| DistilBERT (ours) | 65%, −0.64 | 9%, +0.11 | 0% |
| DistilBERT (HF) | 59%, −0.59 | 15%, +0.15 | 0% |
| RoBERTa-large | 18%, −0.18 | **42%, +0.42** | 0% |
| Logistic regression | 42%, −0.37 | 49%, +0.36 | 2–6% |

**Why.** The control rules out "any odd word means negative": DistilBERT ignores a garbled noun
completely. The shortcut is narrower: when the *sentiment word itself* can't be read, the model
falls back on a default, and the default is negative. That default comes from the training data.
In SST-2, words that DistilBERT's vocabulary has to split into fragments are more common in
negative phrases (16.0% of words, against 10.9% in positive phrases), and phrases containing a
fragmented word are less often positive (49.9% against 61.4%). Coinages such as "unwatchable" and
pieces such as "##ible" (from "horrible" and "terrible") make fragments look negative. A typo
turns "brilliant" into exactly that kind of fragment. Logistic regression behaves differently: it
simply loses the unknown word and drifts towards the middle from both sides (−0.37 / +0.36).

**Connection to the hypotheses.** H1 predicted that typos would expose hidden failures
*including in DistilBERT*. That half of H1 holds (DistilBERT's error is 32 points higher on typo
tests, D47), and this finding explains *why*: not a lack of robustness in general, but a learned
default that only shows when the sentiment word is lost.

*Sources:* `results/typo_direction.csv`, `results/h1_test.csv`; D37, D47.

## 6.3 The shortcut replicates in an independently trained model

**Finding.** Hugging Face's DistilBERT fine-tuned on SST-2 (same architecture, trained by someone
else) matches ours within half a point of accuracy everywhere, and it has the **same typo
shortcut**: 59% flips on positive-word typos, 15% on negative ones, 0% on nouns (Table 6.2).

**Why.** Two separate training runs on the same data learned the same default, so the default is
a property of SST-2, not of one training run. This agrees with the data analysis in 6.2.

**Connection to the hypotheses.** It strengthens H1's typo result: the failure is not an accident
of our model. It also extends H5's conclusion, which showed that the classical models share
their shortcut words, to the transformers: shortcuts come from the training data, and models
trained on the same data inherit the same ones. Within our own DistilBERT, none of the
per-sentence conclusions depends on the random seed: all 291 checks across seeds 13, 42 and 87
agree (171 estimates keep their sign, 120 tests keep their significance; Appendix A).

*Sources:* `results/typo_direction.csv`, `results/seed_robustness.csv`; D44, D46.

## 6.4 Does scale cure shortcuts? It shrinks them

**Finding.** RoBERTa-large (SiEBERT: five times larger than DistilBERT, trained on 15 sentiment
datasets) has the smallest Mirage Gap of all eight models (accuracy 0.946, robustness 0.906) and
passes every negation template pair. But two shortcuts remain:
- **Typos still break it, in the opposite direction.** It flips 42% of *negative* sentences with
  a typo, against 18% of positive ones (Table 6.2): its default for an unreadable sentiment word
  leans positive.
- **Names still move it, slightly.** It flips 1% of name-swap pairs (mean shift +0.010, Holm
  p = 0.02), and across all 35 film people its reaction to a name follows the name's reputation in
  SST-2 (ρ = 0.46, 95% CI 0.06–0.79, on the 23 never-tested names), although the largest effect is
  only ±0.03.

**Why.** More data and parameters let the model read negation in context and rely less on single
words, but its training data still pairs some names and unreadable words with one label.
The default most likely points the other way because its training data, 15 datasets rather than
SST-2 alone, pairs fragmented words with labels differently; we did not test this directly.

**Connection to the hypotheses.** It supports H2 at a larger scale: the transformer advantage on
negation holds and grows. It qualifies H1: hidden failures persist even in a much stronger model;
they get smaller and change direction rather than disappearing.

**Caveat.** SiEBERT was very likely trained on SST-2 and IMDb, so its *accuracy* may be inflated
(Limitations, C12). The behavioural tests use sentences we generated, so the shortcut findings
are not affected by that.

*Sources:* `results/mirage_gap.csv`, `results/typo_direction.csv`, `results/name_effect.csv`,
`results/name_probe.csv`; D45, D46.

## 6.5 Naive Bayes judges a film by who made it

**Finding.** In the name-swap test, changing only a film person's name flips Multinomial NB's
verdict in 38% of pairs (mean shift +0.37, 95% CI 0.34–0.40). To check that this isn't a quirk of
the six tested name pairs, we placed all 35 film people with at least 15 training phrases into
the same 40 neutral sentences. For the 23 names never used in the test, Multinomial NB's reaction
to a name matches how positive that name's training phrases were with ρ = 0.91 (95% CI 0.78–0.96),
and the effects are large (up to ±0.32 in probability). Logistic regression shows a weaker
version (ρ = 0.45, CI 0.10–0.71 over all 35 names; effects up to ±0.13). Both DistilBERTs ignore
names (effects below 0.001).

**Why.** A name carries no sentiment, but in SST-2 each director or actor appears in phrases
with a particular balance of labels. Naive Bayes turns that balance directly into a vote (6.1),
so it has in effect memorised each person's reputation. A contextual model learns that the name
isn't what decides the label.

**Connection to the hypotheses.** It is a hidden failure of exactly the kind H1 describes:
accurate on the test set, but for a reason that breaks on new films. It also illustrates H5's
mechanism: the shortcut words the models share are words whose training labels happen to be
skewed. The correlation is reported with its effect size because a rank correlation of
near-zero effects can look large (HF DistilBERT reaches ρ = 0.51 on effects below 0.0001).

*Sources:* `results/name_effect.csv`, `results/name_probe.csv`, `results/name_probe_by_name.csv`;
D32, D45.

## 6.6 Why negation marking works on templates but not on real sentences

**Finding.** Marking the words after a negation (`not good` → `not NOT_good`) closes 96% (95% CI
93–99%) of logistic regression's negation gap to DistilBERT on templates, but only 23% (CI −2% to
43%, not significant) on the 519 real negated SST-2 sentences. On those real sentences it fixes
38 of the 118 errors but breaks 23 sentences that were right before.

**Why.** An error analysis found three reasons:
1. **Scope swallows whole clauses.** A real negation marks 6.9 words on average, against 2.1 in a
   template, so the signal on `NOT_good` is diluted by `NOT_the`, `NOT_film`, and so on.
2. **Real negation often doesn't negate the sentiment.** Double negation ("it's almost
   impossible not to be moved"), idioms ("it won't be long before…") and negated non-sentiment
   verbs are pushed the wrong way.
3. **Rare marked words have noisy weights.** `NOT_good` (−3.43) is sensible, but `NOT_brilliant`
   comes out positive (+1.07) because it is rare.

A first explanation, that real sentences mark words the model never saw marked, was tested and
rejected: the model knows about as many of them (78% against 74% on templates).

**Connection to the hypotheses.** This explains why **H3 is only partly supported**. Templates
test negation in its easiest form, one negation word right before one sentiment word. Word pairs
and marking together close 40% (CI 21–58%) of logistic regression's real-sentence gap, not all of
it, because real negation is long-range, idiomatic and sometimes double. A contextual model
handles that, which is also H2's result seen from the other side.

*Sources:* `results/negation_gap.csv`, `results/negmark_effect.csv`, LR predictions on
`sst2_neg`; D38, D42.

## 6.7 Being unusual is not the same as being wrong

**Finding.** The embedding distance from the SST-2 training data separates IMDb reviews from SST-2
test sentences well (AUROC 0.93), so it does detect the dataset shift. But *within* IMDb it does
not pick out the reviews a model gets wrong: AUROC 0.49–0.53 for all eight models, no better than
chance. Low confidence does pick them out, for every model (0.55–0.86).

**Why.** Distance measures how unlike the training text an input is: its topic, length and
style. Almost every IMDb review is unlike SST-2's short phrases, so the distance is high for
nearly all of them and says little about which ones are hard. Errors most likely come from the content
(mixed sentiment, sarcasm, a twist at the end), which the model's own confidence reflects and the
distance doesn't.

**Connection to the hypotheses.** This is the reason behind **H4's rejection**: the distance
score works as a shift detector, so the negative result isn't a broken score; it is the wrong
tool for spotting errors. For deployment: monitor distance to know that the data has changed, and
confidence to know which predictions to check.

*Sources:* `results/early_warning.csv` (columns `auroc`, `ood_sanity_auroc`); D25, D30.

---

## 6.8 What the findings have in common

Three points connect them:
- **Shortcuts come from the training data.** Names' reputations (6.5), fragmented words (6.2) and
  their replication across training runs (6.3) all trace back to label imbalances in SST-2.
- **The model family decides how strongly a shortcut is used.** Naive Bayes uses every skewed
  word at full strength; logistic regression shrinks them; transformers mostly read context but
  keep defaults for what they can't read (6.1, 6.2, 6.4).
- **Accuracy hides all of it.** The leaderboard twins (6.1) and the larger model's remaining
  shortcuts (6.4) are invisible without behavioural tests, which is the project's central claim.
