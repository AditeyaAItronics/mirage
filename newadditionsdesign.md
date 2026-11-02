# New additions: design document

Three presentation additions, in build order:

1. **#4: external models.** "Does scale cure shortcuts?"
2. **#1: the Mirage Gap chart**, plus a shortcut fingerprint.
3. **#2: the Shortcut Explorer**, a live demo app.

#4 goes first because the external models become extra dots on the chart (#1) and extra rows in the
demo (#2). Each section gives the goal, what the result looks like (the skeleton), how it will be
built, how it will be tested, and the risks. Every finished part gets an entry in
`majordecisions.md` (D39 onward), and the tables, figures and docs are regenerated.

**Status: built** (Nov 2026). #4 is D39, #1 is D40, #2 is D41 in `majordecisions.md`, which also
records where the build departed from this design:
- IMDb for the external models runs in full on Kaggle, not as a CPU subsample.
- The robustness score uses pair accuracy, not flip rates.
- The demo's Integrated Gradients view was dropped.
- Later (D43): the demo's word importance became the shared `explain.occlusion` function, which
  now also ranks words for H5. The demo's behaviour is unchanged.

**Done:** the Kaggle run is in and checked (D46); every table and figure includes both external
models. One correction came from the results: RoBERTa-large flips on only 1% of name pairs in
the full test, so the demo's Kidman/Seagal talking point and the presenter guide now say scale
*shrinks* the name shortcut rather than "doesn't cure it".

**Nothing left to build here.**

This document is kept as the original design; the decision log is the current record.

---

## Shared rules (apply to all three)

- **Predictions are still the contract.** External models write the same parquet files as every
  other model, so every existing table and figure includes them with no special cases.
- **Nothing changes existing results.** The 18 current models, their predictions and every number in
  D35–D38 stay as they are. New work only adds rows, files and figures.
- **Every new number is traceable.** Each figure is drawn from a CSV in `results/`, and each CSV from
  prediction files.
- **Tests first for the logic.** Pure functions (scores, attribution, label mapping) get unit tests
  that need no model download. The app gets a headless smoke test.

---

## #4: External models ("Does scale cure shortcuts?")

### Goal
Put ready-made, larger sentiment models through *exactly* the same evaluation sets and behavioral
suite, with no training, and answer: do bigger, more broadly trained models still take the
shortcuts we found?

### Models (checked: both exist and download)

| short name | Hugging Face id | size | role |
|---|---|---|---|
| `roberta_large` | `siebert/sentiment-roberta-large-english` | 355M params, 1.36 GB | **Scale.** About 5× DistilBERT, trained on 15 datasets (reviews, tweets…). |
| `distilbert_hf` | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | 66M, 511 MB | **Replication control.** Same architecture as ours, trained on SST-2 by Hugging Face. Do their shortcuts match ours? |

Excluded: `cardiffnlp/twitter-roberta-base-sentiment-latest` has three classes (it adds
neutral) and was trained on tweets; `textattack/roberta-base-SST-2` has no label names in its config.

### Skeleton (what gets added)

```
configs/external.yaml            models: id, short name, batch size, max length, which eval sets
src/mirage/models/external.py    ExternalClassifier: same predict_proba interface as every model
src/mirage/cli.py                new stage `predict-external` (+ registered in `all`)
results/predictions/             roberta_large__base__seed0__*.parquet, distilbert_hf__…
tests/test_external.py           label mapping, contract, CLI stage (with a tiny stand-in model)
```

`ExternalClassifier`:
- Loads the tokenizer and model once and keeps `name`, `max_length` and `long_max_length`.
- **Label mapping comes from the model's own config** (`id2label`): it finds which output index
  means "positive" and which means "negative". It never assumes 0 = negative, and it refuses to
  run if the labels aren't exactly one positive and one negative.
- **Raw text in, no SST-style normalization.** These models were trained on ordinary English; our
  `normalize` ("does n't", "it 's") exists so *our* models share one tokenizer. The eval sets are
  passed exactly as stored. SST-2 text is already in SST's tokenized style, which the external
  models handle somewhat less well; that's documented as a caveat.
- Long text (IMDb): head + tail token truncation at 512, the same rule as our DistilBERT (D23).

### How it will be built
1. `configs/external.yaml` and `ExternalClassifier`, with unit tests using a tiny random
   model built locally (no download), including one with the labels in reverse order.
2. The `predict-external` stage: for each configured model, predict each configured eval set and
   write it through `write_predictions`. It skips sets whose files already exist, like `predict`.
3. Run on the laptop CPU: the behavioral suite (6,250 rows) and `sst2_val`, `sst2_test`, `sst2_neg`
   and `sst2_heldout`. Estimate: RoBERTa-large about 15–25 minutes, `distilbert_hf` about 5 minutes.
4. IMDb, depending on the decision below:
   - (a) a Kaggle cell (full 10,000 + edges, about 10 minutes on GPU), or
   - (b) a 2,000-review stratified subsample on CPU, written as its own eval set so it can't be
     confused with the full one.
5. `evaluate`, `warn` and `figures` re-run. Every table gains the new rows automatically.
   Model-specific tables skip them where they don't apply: `negmark_effect` (no `negmark`
   variant) and IG (only our DistilBERT).
6. `plots.MODEL_ORDER` gains the two names with fixed colours; the palette is checked again for
   eight series.

### Caveats written into the results (not hidden)
- **Possible test contamination:** SiEBERT's card says it was trained on 15 datasets, very likely
  including SST-2 and IMDb. **Its accuracy numbers are therefore not comparable**, because it may
  have seen our test items. **Its behavioral results are clean**, because the test sentences are
  ours. The slides lead with behavioral results for external models.
- `distilbert_hf` was trained on GLUE SST-2 *train*, which includes our held-out phrases. Its
  held-out accuracy is inflated; validation and test are fine.

### Tests
- Label mapping in both orders, and rejection of a three-class model.
- `predict_proba` returns (n, 2) with P(negative), P(positive) and passes `validate_predictions`.
- The stage writes correctly named files (stand-in model, temporary `MIRAGE_ROOT`).

### Risks
- **Download size** (about 1.9 GB together): a one-time download, cached by Hugging Face.
- **CPU memory** for RoBERTa-large (about 2–3 GB): one model at a time, small batches.

**Effort:** about 3–4 hours of build, plus CPU runs.

---

## #1: The Mirage Gap chart and shortcut fingerprint

### Goal
One headline figure for the talk and report: the gap between how good a model *looks* (accuracy)
and how good it *is* (behaving correctly on targeted tests). That gap is the "mirage".

### The robustness score (definition, to be confirmed)
Per model, the average **pass rate** (1 − error) over four core shortcut tests, equally weighted:

| component | source | what it measures |
|---|---|---|
| negation | `beh_negation_dir` | adding "not" flips the model's own answer |
| real negation | `sst2_neg` accuracy | negation in real sentences |
| typo robustness | `beh_typo` | a misspelled sentiment word doesn't flip the answer |
| name robustness | `beh_name_shortcut` | a film person's name doesn't flip the answer |

Not included:
- `typo_noun`, which is a control, not a shortcut test.
- The unscored probes (`negation_neg`, `negation_dir_neg`).
- `name_swap`, whose unseen names make it nearly impossible to fail (D32).

**Sensitivity check:** the model ordering is also computed with each component left out in turn;
the figure caption says whether the ordering holds.

### Skeleton
```
src/mirage/evaluate.py    mirage_gap(preds) -> results/mirage_gap.csv
                          (model, accuracy, robustness, each component, gap = accuracy - robustness)
src/mirage/plots.py       mirage_gap_chart(table)      -> figures/mirage_gap.{png,pdf}
                          shortcut_fingerprint(table)  -> figures/shortcut_fingerprint.{png,pdf}
                          + 16:9 slide versions (…_slide.png)
tests/                    score on hand-made numbers; figures render
```

**Figure 1, the Mirage Gap (scatter):**
- **Axes:** x = SST-2 test accuracy (what a leaderboard shows); y = robustness score (what the model
  actually handles).
- **Points:** one dot per model, directly labelled. Fixed model colours, the same as every other
  figure. External models drawn as hollow dots (different training data).
- **Reference line:** the diagonal "robustness = accuracy". The vertical drop from a dot to the
  diagonal is its mirage, drawn as a thin line, with the largest one annotated
  (e.g. "Naive Bayes: looks 83%, behaves about 50%").
- **Title as the message:** e.g. "Accuracy hides how models fail".

**Figure 2, the shortcut fingerprint (small multiples, not a radar chart):**
- **Layout:** one row per model; one bar per shortcut component, plus calibration (1 − ECE on
  IMDb) and early warning (confidence AUROC).
- **Why bars, not a radar:** radar charts distort areas and are hard to read from the back of a
  room.
- **What it shows:** each model fails in its *own* way. Naive Bayes on names, DistilBERT on
  typos, the classical models on negation.

**Built with the dataviz skill's rules:** colour last and validated; direct labels; no dual axes;
large type for projection; a table view in `mirage_gap.csv`.

**Effort:** about 1.5–2 hours.

---

## #2: The Shortcut Explorer (live demo)

### Goal
Type a sentence and see every model's verdict, plus *why*, in real time. The talk's memorable
moment is watching Naive Bayes flip on "Kidman" vs "Seagal", and DistilBERT flip on "brililant".

### Skeleton
```
app/shortcut_explorer.py   the Streamlit app (UI only; logic imported from mirage)
src/mirage/demo.py         load_models(light), score(text), occlusion(text), flags(text)
app/examples.yaml          the curated example pairs, each with its expected behaviour
app/screenshots/           fallback images of every curated example
tests/test_demo.py         occlusion maths, curated examples behave as expected, headless app run
pyproject.toml             new optional extra: [demo] = streamlit
```

**Screen layout:**
1. **Header:** "Mirage: Shortcut Explorer", and a one-line explanation.
2. **Input:** a dropdown of curated examples (the default) and a free-text box, plus a **Compare**
   switch for two sentences side by side.
3. **Results table, one row per model:**
   - label and probability bar;
   - **word importance:** each word shaded by leave-one-word-out (remove the word, measure the
     change);
   - flags (see below).
4. **Compare mode:** the change in positive probability per model between sentence A and B, with the
   largest changes highlighted. This is where the Kidman/Seagal and brilliant/brililant pairs land.
5. **Optional "detailed view":** Integrated Gradients for our DistilBERT (slower, about 2 seconds).

**Why leave-one-word-out for word importance:**
- It works identically for every model, including the forest, which has no per-word weights.
- It's fast on short sentences (words × models predictions).
- It's explainable in one sentence on stage.

### When the demo meets a sentence the models never learned from (the risk you raised)

**What we measured** (before designing this section):
- **Plain negative words are fine.** "mundane" (18 training uses) and "humdrum" (9) are known,
  and every model scores them clearly negative (0.00–0.03).
- **Sarcasm breaks every model.** "Oh great, another two hours of my life I will never get back."
  gets **0.98 positive** from DistilBERT, 0.52 from LR and 0.10 from Naive Bayes.
- **Unseen words drift.** "a rollercoaster of meh" ("meh" and "rollercoaster": 0 training uses)
  gets 0.80 from DistilBERT, 0.29 from LR and 0.48 from Naive Bayes.

So a live demo *will* sometimes be wrong. The plan is to make that safe and useful, in five layers:

1. **Curated by default, tested in advance.** The dropdown holds about 10 example pairs chosen
   to show each finding, each with its **expected behaviour** written in `app/examples.yaml`
   (e.g. "NB flips; DistilBERT doesn't"). A test checks every curated example still behaves as
   expected, so the scripted part of the demo can't surprise you.

2. **Show what each model can't see.** For the classical models, words outside their training
   vocabulary are drawn **greyed out with a "not in vocabulary" marker**. When "meh" is greyed out,
   a wrong answer explains itself: the model literally can't see the word, which is the
   vocabulary shortcut.

3. **Honest uncertainty flags,** grounded in the project's own results:
   - **"Low confidence"** when the top probability is below 0.70. This is the signal H4 showed
     actually predicts errors (AUROC 0.836 for DistilBERT).
   - **"Unusual input"** when the sentence's embedding distance is above the 95th percentile of
     SST-2 test sentences. It's labelled "unusual", not "probably wrong", because H4 showed
     distance doesn't predict errors.

   A wrong answer with a "low confidence" flag shows the warning working. A wrong answer *without*
   one makes the overconfidence point.

4. **Wrong answers become talking points, not accidents.** The presenter notes include a
   ready line for each kind of failure:
   - **Sarcasm:** "No model here was trained to detect sarcasm. The words say positive, so they say
     positive. That's shortcut learning in its purest form: reacting to words, not meaning." It
     also connects to the limitations slide (sarcasm is listed in shortfalls D4).
   - **Unknown word:** "Greyed out means the model has never seen it. It's guessing from the rest
     of the sentence."
   - **Models disagree:** "This is why we tested six models: their shortcuts are different."

   A curated **"Where everything fails"** example (the sarcasm sentence) is *planned* into the demo,
   so the failure happens on your terms.

5. **Stage safety net:**
   - It never crashes on empty, very long or non-English input (input trimmed to 60 words, with a
     friendly message).
   - **Light mode** leaves out the 822 MB random forest and RoBERTa-large for slow laptops.
   - **Offline:** models load from disk; no internet needed.
   - A **"reset to examples"** button.
   - **Fallback screenshots** of every curated example in `app/screenshots/`, in case the laptop
     or projector misbehaves.

### How it will be built
1. `mirage/demo.py`: model loading (cached), scoring, leave-one-word-out, the vocabulary
   check, and the two flags. Unit tests for each.
2. `app/examples.yaml`, with about 10 curated pairs and their expected behaviour, plus a test
   that runs them.
3. `app/shortcut_explorer.py`: the Streamlit layout only, with no logic of its own.
4. A headless test with Streamlit's `AppTest`: the page loads, a curated example runs, and every
   model returns a probability.
5. A launch entry, so the app opens in the browser pane, and a 1-page **presenter guide**
   (`app/PRESENTING.md`): how to start it, the demo script, and the ready lines.
6. Fallback screenshots generated from the running app.

**Performance target:** about 20–30 seconds to start (once), then under 2 seconds per sentence on
a laptop CPU.

**Effort:** about 4–5 hours.

---

## How each part gets written up

For every part, in this order:
1. **Code and tests,** with the full suite green and lint clean.
2. **Real run,** then **screening**: look at the outputs and figures and check a few raw examples
   by hand, as done for every result so far.
3. **Decision log entry** (D39 external models, D40 Mirage Gap, D41 Shortcut Explorer): what,
   why, what was rejected, and the numbers.
4. **Shortfalls updated:** new limitations (e.g. SiEBERT contamination, the robustness-score
   weights) are added; any resolved ones are moved.
5. **README** (new figure and demo instructions), **codeexplanations.md** (new files) and
   **ARCHITECTURE.md**, then commit and push.

---

## Decisions needed before building

| # | Decision | My recommendation |
|---|---|---|
| 1 | External models: both, or only RoBERTa-large? | **Both.** The replication control costs about 5 minutes of CPU. |
| 2 | IMDb for external models: Kaggle (full, about 15 min of your time) or a 2,000-review CPU sample? | **CPU sample.** IMDb isn't their main story. *(Decided: full 10,000 on Kaggle, for a like-for-like comparison; then every eval set moved to Kaggle.)* |
| 3 | Add Streamlit as an optional `[demo]` dependency? | **Yes.** It doesn't affect the rest of the project. |
| 4 | Robustness score: equal weights over negation (direction), real negation, typo, names? | **Yes,** with the leave-one-out sensitivity check. |
| 5 | Demo flags: confidence threshold 0.70, unusual-input threshold at the 95th percentile? | **Yes.** Both are tunable in the app's config. |
