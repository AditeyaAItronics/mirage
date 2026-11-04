# Confirmatory claims (G8, D53): approved, pre-registered

> **Status: APPROVED by the author on 4 Nov 2026, before any fresh data was run.** The commit that
> adds this file is the pre-registration: its timestamp precedes every prediction on
> `behavioral_v2` and `imdb_fresh`. A claim that fails is reported as failed. Pass rules are judged on the **cluster** confidence interval (G7):
> behavioral cases resampled by adjective, IMDb reviews by film.

**Approved:** 4 Nov 2026, all 17 claims as written, and frame set B including the "Sadly," frame.
**Commit:** the first commit of this file (`git log --follow docs/replication/confirmatory_claims.md`).

## The fresh data
- **`behavioral_v2`:** a new behavioral suite, drawn with seed 2026 from the same word lists and
  name pairs, but with **frame set B** (the paraphrased sentence frames below), so no sentence in it
  appears in the original suite. All eight models.
- **`imdb_fresh`** (and `imdb_fresh_edges`): 10,000 IMDb test reviews that are not in the original
  sample (seed 2026, stratified, capped and cut to edges exactly like `imdb`). All eight models.
  They share some films with the original sample, but no review.
- The comparison set for H1 stays the negation-free SST-2 test sentences: SST-2 has no unused
  labelled sentences.

## Claims

| # | Connects to | Claim | Pass rule (95% cluster CI) | Original value |
|---|---|---|---|---|
| 1 | H1 (typos) | Each transformer's typo-pair failure rate is higher than its plain-sentence error by at least 10 points | lower bound of (typo error − plain error) ≥ 0.10, for all three | +24 to +32 points |
| 2 | H1 (negation) | Logistic regression and Multinomial NB fail more directional-negation pairs than plain sentences | lower bound > 0, for both | +12.2, +41.8 points |
| 3 | H2 | Every single-word classical model fails more directional-negation pairs than DistilBERT, by at least 10 points | lower bound of the gap ≥ 0.10, for all five | +23 to +100 points |
| 4 | H3 (templates) | Word pairs + negation marking close more than half of logistic regression's directional-negation gap | lower bound of ladder_closed > 0.5 | 97% |
| 5 | Findings §2 | DistilBERT flips more positive-sentence typo pairs than negative ones | lower bound of (flip rate positive − flip rate negative) > 0 | 65% vs 9% |
| 6 | Findings §2 | A typo in the noun almost never flips DistilBERT or HF DistilBERT | noun-typo flip rate ≤ 0.02, for both | 0% |
| 7 | Findings §3 | HF DistilBERT shares the typo default | its positive-sentence typo flip rate ≥ 0.30 | 59% |
| 8 | Findings §4 | RoBERTa-large's typo default points the other way | its negative-sentence typo shift > its positive-sentence shift (in absolute value) | 0.42 vs 0.18 |
| 9 | Findings §5 | Changing only a film person's name flips Multinomial NB often and both DistilBERTs almost never | MNB flip rate ≥ 0.20; both DistilBERTs ≤ 0.02 | 38%, 0% |
| 10 | Findings §1 | Multinomial NB behaves worse than logistic regression on the behavioral checks | MNB's mean pass rate over negation, typo and name pairs is lower by ≥ 0.10 | 0.485 vs 0.733 |
| 11 | H4 | Embedding distance does not pick out IMDb errors | every model's distance AUROC ≤ 0.56 | 0.49–0.53 |
| 12 | H4 | Low confidence picks out IMDb errors far better than distance, for the three transformers and LR, RF, MNB | lower bound of (confidence AUROC − distance AUROC) ≥ 0.10, for all six | +0.16 to +0.35 |
| 13 | H4 | Adding distance to confidence adds less than 0.02 AUROC | combined − confidence < 0.02 for every model | ≤ +0.012 |
| 14 | Findings §8 | Full reviews are easier than their first and last sentences | upper bound of (full error − edges error) < 0, for every model | −2.6 to −10.7 points |
| 15 | Findings §8 | At SST-2 length, IMDb text is harder than SST-2 | lower bound of (edges error − SST-2 test error) > 0, for every model | +7.0 to +13.1 points |
| 16 | Findings §12 | RoBERTa-large is at least as accurate on fresh IMDb as on SST-2 test | its fresh-IMDb accuracy ≥ 0.946 | 0.956 |
| 17 | Findings §11 | Multinomial NB is the worst-calibrated of LR, RF, MNB and DistilBERT on IMDb | its ECE is the highest of the four | 0.229 |

## Frame set B (D10): the paraphrased sentence frames

Same slots and word lists as the original frames; only the wording changes. **Please check that
each reads naturally and that none of them adds sentiment of its own.**

**Simple sentences** (also used for the typo and noun-typo tests):
- "In my view, the {noun} was {adj}."
- "Overall, the {noun} felt {adj}."
- "We all agreed the {noun} was {adj}."
- "Frankly, I found the {noun} {adj}."

**Negation (scored: "not <positive>" should be negative):**
- "I did not find the {noun} {adj}."
- "The {noun} wasn't {adj}."
- "Sadly, the {noun} was not {adj}."
- "I can't say the {noun} was {adj}."

**Directional negation pairs (affirmative / negated):**
- "I found the {noun} {adj}." / "I did not find the {noun} {adj}."
- "The {noun} felt {adj}." / "The {noun} didn't feel {adj}."
- "Overall, the {noun} was {adj}." / "Overall, the {noun} wasn't {adj}."
- "I would call the {noun} {adj}." / "I would never call the {noun} {adj}."

**Name shortcut (a film person; the sentiment comes only from the mild adjective):**
- "{name}'s new film was {adj}."
- "I saw the latest {name} {noun} and found it {adj}."
- "This {name} {noun} turned out {adj}."
- "Watched a {name} {noun} last night; it was {adj}."

"Sadly," in the third negation frame is a deliberate check, kept by the author: it is mildly
negative itself, so if a model fails that frame much less than the others, the frame is helping
it. The per-frame table (`behavioral_by_frame.csv`) shows this.

## Outcome (added after scoring; the claims above are unchanged)

Scored on 4 Nov 2026 (D53): **16 of 17 replicated.** Claim 3 (H2) did not: on the second suite
our DistilBERT fails the frame "I would never call the movie great" in 96% of cases, so the
decision tree's and random forest's gaps fall below the 10-point rule. Every number is in
`results/replication.csv`; the explanation is in hypothesisanswers.md (H2) and
interestingfindings.md §15. As planned, the "Sadly," frame (B2) was checked. It does help the
bag-of-words models: on "Sadly, the movie was not great." LR fails 0% (27–31% on B0 and B1),
Multinomial NB 21% (40–44%) and Gaussian NB 60% (84–100%). Our DistilBERT makes no errors on any
negation frame, so it can't show the effect (`results/behavioral_by_frame.csv`).
