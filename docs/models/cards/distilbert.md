# Model card: DistilBERT (fine-tuned by us)

| | |
|---|---|
| **Type** | Transformer language model, fine-tuned for sentiment |
| **Input** | Raw text → normalized → word pieces (30,522-piece vocabulary), up to 128 (IMDb: 512) |
| **Output** | P(positive) from a 2-class head on the [CLS] vector |
| **Size** | 66,955,010 parameters: 6 layers, 768 dimensions, 12 attention heads |
| **Trained on** | SST-2 training split, 3 seeds (Kaggle, 2 × T4 GPU, about 9 minutes each) |
| **Code / file** | `models/distilbert.py` · `artifacts/models/distilbert__base__seed{13,42,87}/` |

Part of the Mirage project ([all models](../README.md)). Every number is from `results/`.

## How it works

DistilBERT is a smaller, faster version of BERT (40% fewer parameters, trained to imitate BERT),
pre-trained on English books and Wikipedia to predict missing words. Text is split into word
pieces ("brilliant" is one piece; an unknown word becomes several, "br ##ili ##lan ##t"). Each
piece becomes a 768-number vector, and 6 transformer layers let every piece look at every other
piece (self-attention, 12 "heads" per layer), so a word's vector reflects its context: "good" after
"not" ends up different from "good" alone. A special [CLS] vector summarizes the sentence; we add a
small classification layer on top and fine-tune the whole network on SST-2.

## Training and hyperparameters

Starting point `distilbert-base-uncased`. Learning rate 2e-5 with linear decay and 6% warm-up,
batch 32 per GPU on 2 GPUs (64 per step), 3 epochs, mixed precision, cross-entropy loss. Evaluated
on the held-out split after each epoch; the best epoch is kept (seeds 13 / 42 / 87 → epochs
3 / 3 / 2; held-out 92.1% / 92.0% / 91.8%, curve in `figures/training_curves.png`). **Not tuned**
(shortfall C3). Per-item analyses use the median seed, 13.

## Performance

| measure | value |
|---|---|
| SST-2 test accuracy / macro F1 | 0.922 / 0.922 |
| Real negated sentences (519) | 0.897 |
| IMDb, 10,000 reviews (other domain) | 0.896 |
| Calibration error (ECE; 0 = perfect), SST-2 test / IMDb | 0.046 / 0.051 |
| How well low confidence flags its IMDb errors (AUROC; 0.5 = chance) | 0.836 |
| Shortcut checks passed: negation pairs / real negation / typo pairs / name pairs | 100% / 90% / 61% / 100% |
| **Robustness / Mirage Gap** (accuracy − robustness) | **0.876 / 0.046** |

*Sources: `results/main_table.csv`, `results/mirage_gap.csv`, `results/early_warning.csv`.*

## Behaviour: what the shortcut tests found

- **Small Mirage Gap (0.046):** passes every negation-template pair and every name pair.
- **Its one big shortcut is typos:** one typo in a positive word flips it to negative in 65% of
  pairs ("brilliant" 0.999 → "brililant" 0.010), because word pieces from unknown words were more
  common in negative training phrases (finding 2). A typo in a noun flips nothing.
- **One negation blind spot:** "I would never call the movie great" is read as positive 96% of
  the time, while every other new negation is read correctly (finding 15).
- **Shares the classical models' sentiment words** to a large degree (0.71–0.77, finding 10),
  including "IMAX".
- **Honest confidence:** well calibrated (ECE 0.046 / 0.051) and its low confidence flags its IMDb
  errors with AUROC 0.836. Distance in its own features also finds them (0.822), but adds nothing
  beyond confidence (H4).

## Intended use and limits

The project's reference model: accurate, well calibrated, context-aware. Not robust to misspelled
sentiment words, and untested beyond English movie reviews. All results are on English movie reviews; sarcasm and other domains are untested.
