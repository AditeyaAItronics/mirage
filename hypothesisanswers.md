# Hypothesis answers: H1–H6 in full

Everything the project found about each hypothesis in the proposal ([proposal.md](proposal.md)):
the exact claim, how it was tested, every relevant number with its confidence interval, the
verdict, *why* the result came out that way, the caveats, and the robustness checks. It is
meant as a source to pick from for the report, not as report text: it is deliberately longer
than any report section will be.

The findings that go beyond the hypotheses are in [interestingfindings.md](interestingfindings.md);
each hypothesis below links to the ones that explain or extend it.

**How to read the numbers.**
- Every number comes from a file in `results/`, named under "Sources". "DNN" refers to the entry
  in [majordecisions.md](majordecisions.md) that explains the analysis.
- Brackets are 95% bootstrap confidence intervals (1,000 resamples). Since D52 (G7) they
  resample **clusters**, not single items: template cases that share their sentiment adjective are
  drawn together, and so are IMDb reviews of the same film. SST-2 sentences are their own clusters.
  This only widened intervals (median 1.0–1.5×); no estimate and no H1–H6 verdict changed
  (`results/cluster_robustness.csv` lists both versions).
- "Meaningful" (D52, G9) means the interval also clears a pre-set minimum effect
  (`configs/stats.yaml`: 2 points of error or probability, 0.02 AUROC, 10% of a gap closed).
- **Replication (D53, G8):** 17 of these claims were written down before fresh data was scored
  ([docs/confirmatory_claims.md](docs/confirmatory_claims.md)) and re-tested on 10,000 IMDb
  reviews never used before and on a second template suite with new sentence frames (frame set B).
  16 replicated, 1 did not (H2, see below). Each hypothesis's section ends with its replication.
- "Holm p" is the p-value after Holm's correction within one family of tests (D28). Our models,
  the two external models (D39) and the single-word ablation (D42) are separate families, so
  adding models never changed an earlier result.
- Models: logistic regression (LR), Multinomial NB (MNB), Gaussian NB (GNB), random forest (RF),
  decision tree (DT), our DistilBERT (median of seeds 13 / 42 / 87), and two ready-made external
  models: Hugging Face's DistilBERT (HF DistilBERT) and SiEBERT RoBERTa-large.
- Classical variants: `unigram` (single words), `base` (single words + word pairs), `negmark`
  (base + negation marking up to punctuation), `negmark_w3` (marking at most 3 words).
- Eval sets: SST-2 test (1,820 sentences, public labels), the real negation subset `sst2_neg`
  (519 SST-2 validation + test sentences containing a negation), IMDb (10,000 reviews, capped to
  the first + last words), `imdb_edges` (each review's first and last sentence), and the
  behavioral suite (3,750 template cases).

## Summary

| | Hypothesis (short) | Verdict | Key number |
|---|---|---|---|
| **H1** | Every model, DistilBERT included, errs more on negation and typos | **Partly supported** | Typos: clearly higher for 6 of 8 models (DistilBERT +32 points). Negation: higher for the bag-of-words models, not for the transformers |
| **H2** | DistilBERT handles negation much better than single-word bag-of-words models | **Supported**, with a replication caveat | 14–35 points better on real negated sentences, 9–100 on templates, all Holm p < 0.001. On new frames DistilBERT fails "would *never* call it great", and the tree's gap vanishes |
| **H3** | Word pairs + negation marking close part, not all, of the gap | **Supported for LR**; on templates for every model | LR, real sentences: 40% [21, 58] closed. Templates: 25–98% |
| **H4** | Embedding distance adds information beyond confidence | **Rejected** (replicated) | Distance AUROC 0.49–0.53 vs confidence 0.55–0.86; combining adds nothing (one tiny exception). In DistilBERT's *own* features distance reaches 0.82, but still adds nothing to confidence |
| **H5** | Classical models share their shortcut words | **Supported for LR, RF, MNB**; not GNB or DT | Reliability-corrected agreement 0.88–0.92 |
| **H6** | Gaussian NB is worse than Multinomial NB | **Supported** | Accuracy 59.1% vs 83.4%; F1 0.589 vs 0.833; McNemar p ≈ 10⁻⁵⁶ |

---

## H1 — Hidden failures on negation and typos

> **H1 (proposal):** Every model, including DistilBERT, has a clearly higher error rate on
> negation and sentiment-word typos than on the normal test set.

### How it was tested (D47)
- **Comparison set:** the SST-2 test sentences *without* any negation (1,481 sentences). It is
  disjoint from the negation sets, so negation and non-negation are compared on different items.
  Using the whole test set would have put the negated sentences on both sides.
- **Tests:** the real negated sentences (`sst2_neg`, 519), two negation template tests
  (`beh_negation`: "The film was good." vs "The film was not good."; `beh_negation_dir`: the
  directional version, scored as a pair), and the typo test (`beh_typo`, 500 pairs, a typo in the
  sentiment adjective).
- **Statistic:** error rate on the test minus error rate on the comparison set. Because the two
  sets are different items, each is resampled on its own (`stats.diff_independent_ci`).
- **"Clearly higher"** = the difference is above zero and significant after Holm.

### Results

**Table H1. Error-rate difference vs negation-free SST-2 test sentences, in points [95% CI].
✔ = clearly higher (Holm p < 0.05).**

| model | error on comparison set | real negation | negation templates | directional negation | typos |
|---|---|---|---|---|---|
| Logistic regression | 15.5% | **+7.3** [3.5, 11.5] ✔ | **+11.7** ✔ | **+12.2** ✔ | **+29.9** [25.3, 34.7] ✔ |
| Multinomial NB | 15.5% | **+8.4** [4.5, 12.7] ✔ | **+33.7** ✔ | **+41.8** ✔ | **+23.1** ✔ |
| Gaussian NB | 39.2% | **+8.6** [3.3, 13.7] ✔ | **+54.2** ✔ | **+56.6** ✔ | −25.4 |
| Random forest | 23.8% | +3.6 [−0.5, 7.7] | −3.2 | +1.0 | **+12.8** ✔ |
| Decision tree | 33.8% | +5.5 (Holm p 0.11) | −25.8 | −13.3 | −10.8 |
| DistilBERT (ours) | 7.4% | +2.6 [−0.4, 5.4] | −7.4 (0 errors) | −7.4 (0 errors) | **+32.0** [20.1, 42.4] ✔ |
| HF DistilBERT (external) | 7.3% | **+3.5** [0.5, 6.5] ✔ (Holm p 0.044) | −7.3 (0 errors) | −7.3 (0 errors) | **+31.7** ✔ |
| RoBERTa-large (external) | 5.3% | +1.6 [−1.0, 4.1] | −5.3 (0 errors) | −5.3 (0 errors) | **+23.9** [15.9, 32.6] ✔ |

The typo intervals widened most under clustering (DistilBERT's from ±4.5 to about ±11 points):
DistilBERT's typo errors concentrate on some adjectives more than others, so the 500 typo pairs
behave like fewer independent pieces of evidence. Every ✔ survives. The decision tree's −10.8
is no longer significant (Holm p 0.09), which doesn't change its reading.

### Verdict: partly supported
- **The typo half holds, including for DistilBERT.** Every model that actually reads sentiment
  words errs much more when the sentiment word is misspelled: LR +30, MNB +23, RF +13, and all
  three transformers +24 to +32 points. DistilBERT's typo error (39.4%) is 5.3 times its normal
  error.
- **The negation half holds for the bag-of-words models**: LR and MNB are clearly worse on all
  three negation tests, GNB too.
- **It fails for the transformers.** They make *no* template negation errors, and on real
  negated sentences our DistilBERT is only 2.6 points worse (not significant). Only HF DistilBERT
  is significantly worse, by a small 3.5 points. So "including DistilBERT" does not hold for
  negation. This is the same result as H2, seen from the other side.
- **The weakest models do *better* on typos than on normal sentences** (DT −10.8, GNB −25.4).
  This doesn't mean they are robust: they barely react to the sentiment word in the first place,
  so losing it costs them nothing (see [interestingfindings.md §9](interestingfindings.md#9-a-model-that-cant-read-cant-be-fooled)).

### Why
- **Typos:** a bag-of-words model simply loses the misspelled word (it is outside its
  vocabulary), so the sentence's main sentiment cue disappears. DistilBERT doesn't lose it; it
  splits it into fragments, and it has learned that fragments mean "negative"
  ([interestingfindings.md §2](interestingfindings.md#2-distilberts-default-an-unreadable-sentiment-word-means-negative)).
- **Negation:** a bag-of-words model with single words sees "good" in "not good" and counts it
  as positive; word pairs and marking help only partly (H3). A transformer reads "not good" as a
  phrase.

### Caveats
- The template tests use short, clean sentences; the real negation subset is the more realistic
  test of the two.
- The typo test is a template test (one misspelled adjective per sentence), so its error is not
  directly a real-world typo rate.
- H1 was formally tested late in the project (D47), after the other results were known. The test
  and comparison set follow the proposal's wording exactly, but the analysis code was written
  after the data was seen. The replication below answers this: its claims were fixed first.

### Robustness checks
- **Every DistilBERT seed:** the conclusions hold on seeds 13, 42 and 87 (D44, `seed_robustness.csv`).
- **Test-only negation:** restricting the real negation set to the 339 test sentences (never used
  to pick DistilBERT's seed) gives the same signs (D31).

### Replication on fresh data (D53): both claims replicated
- **Claim 1** (transformers' typo error ≥ 10 points above plain): on the second suite, DistilBERT
  +33.8 (lower bound +22.7), HF DistilBERT +34.7 (+24.4), RoBERTa-large +27.1 (+18.2).
- **Claim 2** (LR and MNB fail more directional-negation pairs than plain sentences): LR +36.7
  (lower bound +28.7), MNB +43.0 (+31.6), larger than on the original suite.

**Connected findings:** [§2 the typo default](interestingfindings.md#2-distilberts-default-an-unreadable-sentiment-word-means-negative),
[§3 its replication](interestingfindings.md#3-the-shortcut-replicates-in-an-independently-trained-model),
[§1 leaderboard twins](interestingfindings.md#1-leaderboard-twins-the-same-accuracy-very-different-behaviour),
[§9 a model that can't read can't be fooled](interestingfindings.md#9-a-model-that-cant-read-cant-be-fooled).

*Sources:* `results/h1_test.csv`, `results/errors_by_type.csv`; D47, D31, D44.

---

## H2 — DistilBERT handles negation much better

> **H2 (proposal):** DistilBERT handles negation much better than the bag-of-words models with
> single-word features.

### How it was tested (D42)
- The **single-word (`unigram`) variant** of every classical model, as the proposal names it.
  The main classical models use word pairs as well, so a separate unigram variant was trained
  (D42); testing H2 on the word-pair models would have tested a different claim.
- **Negation gap** = classical model's error − DistilBERT's error, on the *same* items, so the
  bootstrap is paired.
- Four tests: real negated sentences (519), real negated test sentences only (339), the negation
  template, and the directional negation template.
- Holm over the 20 gaps (5 models × 4 tests). External models are excluded from H2: it is about
  our DistilBERT.

### Results

**Table H2. Negation gap: single-word classical error − DistilBERT error, in points [95% CI].
All Holm p < 0.001.**

| model (single words) | real negated sentences (519) | test sentences only (339) | negation template | directional template |
|---|---|---|---|---|
| Logistic regression | **+16.4** [12.3, 20.6] | +15.9 | +32.0 [26.5, 36.9] | +30.9 |
| Multinomial NB | **+14.1** [10.2, 18.3] | +12.4 | +64.2 | +66.0 |
| Random forest | **+17.9** [13.7, 22.0] | +20.6 | +25.6 | +26.8 |
| Decision tree | **+24.3** [19.8, 28.9] | +25.4 | +8.8 | +23.3 |
| Gaussian NB | **+34.7** [29.7, 39.7] | +32.2 | +95.2 | +99.6 |

DistilBERT's own error: 10.0% on the real negated sentences, 0% on both templates.

### Verdict: supported
Every single-word classical model is significantly worse than DistilBERT on negation, on every
test: by 14–35 points on real negated sentences and by 9–100 points on templates. "Much better"
holds: on real sentences DistilBERT makes 10% errors where single-word LR makes 26%.

### Why
A single-word model has no way to represent "not good": it sees the words "not" and "good" as
separate, unordered features. "good" pushes positive whatever comes before it, and "not" can only
add a fixed negative push, which is wrong for "not bad". DistilBERT's self-attention combines
each word with its context, so "not" can reverse the meaning of the word it modifies. The
template results show this most clearly: DistilBERT makes no errors at all, while Gaussian NB,
which cannot even weigh individual words well (H6), fails nearly every case.

### Caveats
- DistilBERT is not tuned (C3) and has 3 seeds (C4); the classical models are tuned. That works
  *against* H2, so the result is, if anything, conservative.
- Part of the real negation set (the 180 validation sentences) was used to choose DistilBERT's
  median seed; the test-only column (339 sentences) avoids this and gives the same result (D31).
- RoBERTa-large confirms the pattern at a larger scale (0 template errors, 6.9% real-sentence
  errors), but it is not part of the H2 test.

### Replication on fresh data (D53): not replicated, and why
**Claim 3** said every single-word classical model is at least 10 points worse than DistilBERT on
the directional negation template (lower bound ≥ 0.10, all five). On the second suite (new
sentence frames, frame set B) it held for three of five models:

| model (single words) | original suite | second suite |
|---|---|---|
| Gaussian NB | +99.6 | +77.5 (lower bound +74.4) ✔ |
| Multinomial NB | +66.0 | +61.7 (+53.0) ✔ |
| Logistic regression | +30.9 | +26.1 (+19.5) ✔ |
| Random forest | +26.8 | +14.3 (**+4.5**) ✘ |
| Decision tree | +23.3 | **+0.4** (+0.0) ✘ |

**Why it failed:** the classical models did not get better. DistilBERT got worse. The new frame
B3 ("I would call the movie great." → "I would **never** call the movie great.") negates with
*never*, and our DistilBERT reads the negated sentence as positive in **96%** of the cases
[90, 100] (`behavioral_by_frame.csv`); HF DistilBERT fails 99%. Ours makes no errors in the
other three frames (B0–B2). RoBERTa-large fails only 7% [0, 17] of the B3 cases, so the
sentence is readable: the gap is in the two DistilBERTs. The decision tree's gap collapses for a second reason:
the directional test only scores a pair when the affirmative sentence is right, and the tree
gets "I would call the movie great" wrong (p = 0.26), so only 8 of its B3 pairs count. Its
failure rate falls from 20% to 3% because the hard cases are filtered out.

**What it means for H2.** "DistilBERT handles negation much better" holds for *not*-negation
and on real sentences; it does not hold for every negation construction. Our DistilBERT reads
every other new negation right ("I did not find the movie good", "The movie wasn't great",
"I can't say the movie was good": 0 errors), but not "I would never call the movie great". Both
DistilBERTs were fine-tuned on SST-2, which suggests that construction is rare there; one frame
is not enough to say whether *never* in general is the problem. The claim is reported as not replicated, as
pre-registered; the original-suite numbers above stand, but the report should say H2 is
supported *for the negation the training data shows most*.

**Connected findings:** [§4 scale](interestingfindings.md#4-does-scale-cure-shortcuts-it-shrinks-them),
[§6 negation marking](interestingfindings.md#6-why-negation-marking-works-on-templates-but-not-on-real-sentences).

*Sources:* `results/negation_gap.csv` (columns `gap_unigram*`), `figures/negation_gap_sst2_neg.png`; D42, D31.

---

## H3 — Word pairs and negation marking close part of the gap

> **H3 (proposal):** Word-pair features and negation marking close part, but not all, of that gap.

### How it was tested (D42, D38)
A **feature ladder** for each classical model, all measured against DistilBERT on the same items:
1. single words (`unigram`): the H2 gap;
2. + word pairs (`base`);
3. + negation marking (`negmark`: every word after "not", "n't", "no", "never"… up to the next
   punctuation gets a `NOT_` prefix, Pang et al. 2002).

- **ladder_closed** = (gap with single words − gap with pairs + marking) / gap with single words:
  the share of H2's gap that both additions together remove. H3 predicts this is above 0
  ("part") and below 1 ("not all").
- **pairs_closed** = the share removed by word pairs alone; **gap_closed** = the share removed by
  marking on top of pairs.
- Each rung is computed on the items it shares with the other rungs, with a paired bootstrap.

### Results

**Table H3a. Share of the single-word negation gap closed by word pairs + negation marking
(ladder_closed) [95% CI], with Holm p.**

| model | real negated sentences (519) | test sentences only (339) | negation template | directional template |
|---|---|---|---|---|
| **Logistic regression** | **40%** [21, 58], p < 0.001 | 43% [18, 65], p 0.004 | **98%** [94, 100] | **97%** [94, 99] |
| Multinomial NB | 18% [−12, 42], p 0.98 | 2% [−47, 35] | 77% [67, 86] | 71% [59, 81] |
| Random forest | 1% [−21, 19], p 1.0 | 11% [−11, 31] | 57% [39, 70] | 51% [23, 65] |
| Decision tree | −27% [−54, −6], p 0.12 | −37% | 77% [58, 87] | 77% [54, 89] |
| Gaussian NB | 18% [5, 30], p 0.04 | 9% [−10, 25] | 25% [17, 33] | 32% [25, 37] |

**Table H3b. LR's ladder, real negated sentences: gap to DistilBERT in points.**

| rung | gap | share of the single-word gap closed |
|---|---|---|
| single words | +16.4 [12.3, 20.6] | — |
| + word pairs | +12.7 [8.9, 16.6] | 22% [4, 39] (pairs alone) |
| + negation marking | +9.8 [6.2, 13.3] | **40%** [21, 58] (both) |

On top of word pairs, marking alone closes 23% [−2, 43] of LR's remaining real-sentence gap (not
significant) but 96% [93, 99] on the directional template.

### Verdict: supported for logistic regression; on templates for every model
- **Logistic regression is the clean case of H3:** on real negated sentences, word pairs and
  marking together close 40% of its gap to DistilBERT, significantly more than 0 and clearly
  less than 100%. "Part, but not all" holds exactly. The test-only sentences agree (43%).
- **On templates, every model closes part of its gap:** 25–98%, all significant, none at 100%.
- **On real sentences, the other models barely benefit.** MNB, RF and DT show no significant gain;
  GNB a small one (18%, Holm p 0.04). The decision tree even gets slightly worse.

### Why the gain is large on templates and small on real sentences
An error analysis of LR (D38, [interestingfindings.md §6](interestingfindings.md#6-why-negation-marking-works-on-templates-but-not-on-real-sentences)):
templates contain negation in its easiest form, one negation word right before one sentiment
word, which is exactly what marking captures. Real negation marks 6.9 words on average (2.1 in
templates), is often double ("almost impossible not to be moved") or idiomatic, and marked rare
words get noisy weights. Marking fixes 38 of LR's 118 real-sentence errors but breaks 23 sentences
it had right.

Why LR benefits and the others don't (a likely explanation, not tested directly): LR learns a
separate weight for `NOT_good` that is roughly the opposite of `good`'s (−3.43, D38). MNB smooths
rare marked words towards an uninformative estimate, and the tree models split on only a few
features, so the many new marked features rarely get used.

### Caveats
- The ratio is unstable when the single-word gap is small, hence the wide CIs for MNB and RF.
- The scope rule is a choice. A second rule (`negmark_w3`: at most 3 words, skipping fixed phrases
  like "not only") gives the same conclusion: big template gains, small real-sentence gains (D31).

### Replication on fresh data (D53): replicated
**Claim 4** (word pairs + marking close more than half of LR's directional-negation gap): on the
second suite the ladder closes **141%** of the gap (lower bound 128%). Over 100% means LR with
pairs and marking makes *fewer* directional-negation errors than DistilBERT there. It is the same
"never call it great" frame as in H2: DistilBERT fails it, while negation marking turns
"never call" and "great" into negated features. The claim's threshold is met with room to spare,
but on the second suite the number says as much about DistilBERT's blind spot as about H3.

**Connected findings:** [§6 negation marking](interestingfindings.md#6-why-negation-marking-works-on-templates-but-not-on-real-sentences),
[§13 marking makes the tree unstable](interestingfindings.md#13-negation-marking-makes-the-decision-tree-unstable-on-names).

*Sources:* `results/negation_gap.csv` (`ladder_closed`, `pairs_closed`, `gap_closed`),
`results/negmark_effect.csv`; D38, D42, D31.

---

## H4 — Does embedding distance predict errors beyond confidence?

> **H4 (proposal):** Gaussian distance on sentence embeddings adds information beyond model
> confidence for predicting errors on IMDb.

### How it was tested (D25, D30)
- **Distance:** each IMDb review is split into sentences and embedded with MiniLM; a Gaussian
  (Mahalanobis, Ledoit-Wolf covariance) is fitted to the SST-2 training sentences; a review's score
  is its *most unusual* sentence's distance.
- **Confidence:** 1 − the model's top probability ("low confidence").
- **Combined:** a logistic regression on the ranks of both scores, fitted out-of-fold on IMDb
  (each review is scored by a model that didn't see it).
- **Measure:** AUROC for picking out the reviews the model gets wrong (10,000 reviews). H4 holds if
  the combined score beats confidence alone. Distance vs confidence and combined vs confidence are
  paired bootstrap differences, Holm over models.
- **Controls:** review length as a baseline predictor, and a length-adjusted distance (D30).
- **Sanity check:** distance must separate IMDb from SST-2 at all, or a negative result would only
  mean the distance is broken.

### Results

**Table H4. AUROC for detecting each model's IMDb errors [95% CI]. Sanity check: distance separates
IMDb from SST-2 test sentences with AUROC 0.926.**

| model | IMDb errors | distance | low confidence | combined | combined − confidence |
|---|---|---|---|---|---|
| DistilBERT (ours) | 1,004 | 0.530 [0.512, 0.550] | **0.836** [0.822, 0.850] | 0.835 | −0.001 |
| HF DistilBERT | 1,073 | 0.525 | **0.845** | 0.844 | −0.000 (n.s.) |
| RoBERTa-large† | 444 | 0.507 | **0.857** [0.836, 0.877] | 0.857 | −0.000 (n.s.) |
| Logistic regression | 2,046 | 0.518 | **0.751** | 0.752 | +0.001 (n.s.) |
| Random forest | 2,969 | 0.521 | **0.702** | 0.703 | +0.001 (n.s.) |
| Multinomial NB | 2,399 | 0.530 | **0.688** | 0.700 | **+0.012** [0.006, 0.018], Holm p 0.003 |
| Decision tree | 3,877 | 0.512 | **0.562** | 0.559 | −0.003 (n.s.) |
| Gaussian NB | 4,542 | 0.493 | **0.547** | 0.549 | +0.001 (n.s.) |

Distance is significantly *worse* than confidence for every model (Holm p < 0.001). Length does as
badly as distance (0.49–0.54), and adjusting distance for length doesn't help (0.49–0.53).

### Verdict: rejected
Embedding distance does not predict which IMDb reviews a model gets wrong (AUROC ≈ 0.5, chance
level, for all eight models), and adding it to confidence adds nothing for seven of eight models.
The one exception is Multinomial NB, where the combination gains 1.2 AUROC points: statistically
significant but practically negligible, and probably due to MNB's poorly spread confidence (its
probabilities are saturated near 0 and 1, D26), which leaves distance a little room to break ties. Confidence, by contrast, is a useful warning signal
for every model, and the better the model, the better the signal (0.55 for GNB to 0.86 for
RoBERTa-large).

### Why
The distance works as designed: it separates IMDb from SST-2 (0.926). But it measures how unlike
the *training text* an input is (topic, length, style), and nearly every IMDb review is equally
unlike SST-2's short phrases. So it says "this is IMDb" for almost every review and nothing about
which reviews are hard. A model's errors most likely come from the content (mixed sentiment,
sarcasm, a twist at the end), which its own confidence partly reflects and an external embedding
doesn't see. The embedding model (MiniLM) is also a different model from the classifier, so its
idea of "unusual" need not match the classifier's: the F2 test below measures it in the
classifier's own features.

### Caveats
- One embedding model and one distance were used; the distance method was chosen on SST-2-vs-IMDb
  separation, not on error detection (D25), which is the fair way round.
- The test is on whole-review shift (SST-2 → IMDb). Distance may still help for errors on inputs
  far outside any training domain.

### Distance in the classifier's own features (F2, D53)
The test above measures "unusual" with an outside embedding (MiniLM). Lee et al. measure it in
the classifier's *own* features instead. `own-distance` does that: a class-conditional
Mahalanobis distance, fitted on SST-2 training, in DistilBERT's [CLS] features (after its
pre-classifier layer) and, for each classical model, in its own TF-IDF features reduced to 300
dimensions (`results/early_warning_own.csv`, the main 10,000 IMDb reviews).

| model | own-feature distance | low confidence | combined − confidence |
|---|---|---|---|
| DistilBERT (ours) | **0.822** [0.809, 0.837] | 0.836 | +0.000 (Holm p 1.0) |
| Logistic regression | 0.466 | 0.751 | −0.000 |
| Random forest | 0.498 | 0.702 | +0.002 (not meaningful) |
| Multinomial NB | 0.533 | 0.688 | **+0.034** [0.026, 0.041], meaningful |
| Gaussian NB | 0.465 | 0.547 | +0.019 (below the 0.02 minimum) |
| Decision tree | 0.486 | 0.562 | −0.006 |

- **For DistilBERT the distance becomes informative**: 0.82, against 0.53 with MiniLM. In its own
  features, the reviews it gets wrong really are unusual. But they are the same reviews its
  confidence already flags, so combining the two adds exactly nothing. This is the stronger form
  of the H4 rejection: even a distance that does detect errors carries no information beyond
  confidence.
- **For the bag-of-words models it stays at chance** (0.47–0.53): TF-IDF features don't say
  which reviews are hard.
- **Multinomial NB is the exception again**, now larger (+3.4 points, above the 0.02 minimum).
  The cause is the one given above: its confidence is saturated near 0 and 1 (D26), so many
  reviews tie on confidence, and almost any second signal breaks ties. It is a weakness of MNB's
  confidence, not a strength of distance.

### Replication on fresh data (D53): all three claims replicated
On 10,000 fresh IMDb reviews, with every model including the Kaggle-run external ones:
- **Claim 11** (distance AUROC ≤ 0.56, every model): 0.50–0.53.
- **Claim 12** (confidence beats distance by ≥ 0.10, lower bound, for the transformers, LR, RF,
  MNB): +0.17 to +0.35, every lower bound ≥ +0.15.
- **Claim 13** (combining adds < 0.02 AUROC, every model): −0.006 to +0.006. MNB's gain from
  the main sample (+0.012) halves to +0.006 (not significant), which supports reading it as
  noise around zero.

**Connected findings:** [§7 unusual is not wrong](interestingfindings.md#7-being-unusual-is-not-the-same-as-being-wrong),
[§11 calibration](interestingfindings.md#11-confidence-is-honest-for-some-models-and-not-others).

*Sources:* `results/early_warning.csv`, `results/early_warning_deciles.csv`,
`results/early_warning_sanity.csv`, `figures/error_vs_score_*.png`; D25, D26, D30.

---

## H5 — Do the classical models share their shortcut words?

> **H5 (proposal):** Classical models share many of the same top shortcut words, suggesting the
> shortcuts come from the data.

### How it was tested (D43)
- **One rule for every model:** a word's effect = the drop in P(positive) when that word is removed
  from a sentence (leave one word out), averaged over the SST-2 test sentences containing it. Every
  model is measured the same way, so the forest and the tree are comparable to LR. Earlier, each
  model's own explanation (coefficients, feature importance) was compared, which mixed up method
  and model (D16, E2).
- **Agreement:** the Spearman correlation of word effects between two models over the 773 words seen
  at least 5 times, with a bootstrap CI.
- **Reliability correction:** each model's split-half reliability (how reproducible its own word
  effects are across two halves of the data, Spearman-Brown) caps how high its agreement with
  anyone can be. The corrected correlation divides by that cap, so a noisy model isn't mistaken for
  a different one.
- Secondary view: overlap of each pair's top-50 words (Jaccard, rank-biased overlap).

### Results

**Table H5a. Agreement of word effects between models (773 words): raw Spearman ρ [95% CI] and
reliability-corrected.**

| pair | raw ρ | corrected |
|---|---|---|
| **LR – Multinomial NB** | 0.76 [0.72, 0.79] | **0.92** |
| **LR – random forest** | 0.79 [0.75, 0.82] | **0.89** |
| **Random forest – Multinomial NB** | 0.70 [0.66, 0.74] | **0.88** |
| Decision tree – LR / RF / MNB | 0.22–0.35 | 0.44–0.64 |
| Gaussian NB – anyone | 0.09–0.22 | 0.18–0.26 |
| DistilBERT – LR / RF / MNB | 0.44–0.53 | 0.71–0.77 |

**Table H5b. Split-half reliability of each model's own word effects.** LR 0.91, random forest
0.86, Gaussian NB 0.77, Multinomial NB 0.75, DistilBERT 0.52, decision tree 0.34.

### Verdict: supported for the three main classical models
LR, random forest and Multinomial NB agree on what each word does almost as much as their own noise
allows (0.88–0.92 corrected). Three models with different learning rules (a linear model, an
ensemble of trees, a generative model) rely on the same words in the same direction, which points
to the shortcuts coming from the training data, as H5 predicts. Gaussian NB and the decision tree
are exceptions.

### Why the exceptions
- **Gaussian NB** reads words only through 300 compressed SVD components, so a single word's effect
  is smeared across components; removing one word changes little and in no consistent direction.
  This is also why it is the weakest model (H6).
- **The decision tree** splits on only a few words; its word effects are the least reproducible of
  all (reliability 0.34), so even after correction it agrees only moderately.

### Caveats
- Removing one word ignores interactions between words (E4).
- The 5-occurrence threshold is a choice; at 10 occurrences (352 words) the three main models agree
  even more (0.91–0.93 corrected; D43).
- The top-50 overlap (secondary view) is lower and noisier, as expected for rankings of 773 words.

**Connected findings:** [§3 the replication](interestingfindings.md#3-the-shortcut-replicates-in-an-independently-trained-model),
[§5 names](interestingfindings.md#5-naive-bayes-judges-a-film-by-who-made-it),
[§10 DistilBERT shares the words too](interestingfindings.md#10-distilbert-shares-the-classical-models-sentiment-words).

*Sources:* `results/explain/occlusion_agreement.csv`, `results/explain/occlusion_reliability.csv`,
`results/explain/overlap_occlusion.csv`, `figures/h5_word_effect_agreement.png`; D43, D16.

---

## H6 — Gaussian NB vs Multinomial NB

> **H6 (proposal):** Gaussian Naive Bayes performs worse than Multinomial Naive Bayes, because a
> Gaussian assumption fits text features poorly.

### How it was tested
Accuracy and macro F1 of both models on every eval set, plus McNemar's test on the paired
predictions (same items), with Holm across all model pairs.

### Results

**Table H6. Gaussian NB vs Multinomial NB.**

| eval set | GNB accuracy | MNB accuracy | GNB macro F1 | MNB macro F1 | only MNB right / only GNB right | McNemar Holm p |
|---|---|---|---|---|---|---|
| SST-2 test (1,820) | 0.591 | **0.834** | 0.589 | **0.833** | 605 / 163 | ≈ 6 × 10⁻⁵⁶ |
| SST-2 validation (872) | 0.569 | **0.805** | 0.568 | **0.804** | 297 / 91 | ≈ 3 × 10⁻²⁴ |
| Real negated sentences (519) | 0.522 | **0.761** | 0.508 | **0.742** | 183 / 59 | ≈ 3 × 10⁻¹⁴ |
| IMDb (10,000) | 0.546 | **0.760** | 0.541 | **0.752** | 3,257 / 1,114 | < 10⁻²²⁰ |
| IMDb edges (10,000) | 0.519 | **0.702** | 0.519 | **0.698** | 3,395 / 1,565 | < 10⁻¹⁴⁰ |

### Verdict: supported
Gaussian NB is 18–24 points less accurate than Multinomial NB on every eval set, and McNemar's test
is overwhelming on all five. On SST-2 test, MNB is right where GNB is wrong in 605 sentences, the
reverse in only 163. Gaussian NB is close to chance (majority-class rate ≈ 50%).

### Why
Text features are sparse, non-negative counts: most words are absent from most sentences, and a
present word is usually present once. Multinomial NB models exactly that, the probability of each
word given the class. Gaussian NB assumes each feature is a bell curve around a class mean, which
fits neither a count that is almost always 0 nor the compressed features it has to use instead: it
needs dense input, so it reads 300 SVD components of the word features rather than the words
themselves. Its class-conditional bell curves on those components overlap heavily, so it separates
the classes poorly. The same compression explains why it doesn't share the other models' word
effects (H5) and why it fails nearly all negation templates (H1, H2).

### Caveats
- The comparison is "Gaussian NB on SVD features" vs "Multinomial NB on word counts", so it tests the
  Gaussian assumption together with the compression it forces. The proposal's wording ("a Gaussian
  assumption fits text features poorly") covers this, but the report should say that the two can't
  be separated here (C6).
- GNB is weak enough that a reader might call it a strawman; it is the comparison H6 asks for, not a
  model we expected to compete (C9).

**Connected findings:** [§1 leaderboard twins](interestingfindings.md#1-leaderboard-twins-the-same-accuracy-very-different-behaviour),
[§9 a model that can't read can't be fooled](interestingfindings.md#9-a-model-that-cant-read-cant-be-fooled).

*Sources:* `results/main_table.csv`, `results/mcnemar.csv`; C6, C9.
