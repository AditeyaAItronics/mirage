# Presenting the Shortcut Explorer

## Before the talk (once, with internet)
1. Install the demo extra: `.venv/Scripts/python -m pip install -e ".[dl,demo]"`
2. Start it once so every model is cached:
   `.venv/Scripts/python -m streamlit run app/shortcut_explorer.py`
   It opens at http://localhost:8501. The first start takes 20–40 seconds; after that, each
   sentence takes about a second.
3. From then on it works **offline**: models load from disk.
4. If the laptop is slow, switch on **Light mode** in the sidebar. It leaves out the random forest
   and RoBERTa-large.
5. Fallback images of every curated example are in `app/screenshots/`. Regenerate them with
   `python app/make_fallbacks.py` if the models change.

## Demo script (about 4 minutes)
Use the **Curated examples** dropdown, in this order. Every expected behaviour below is checked by
`tests/test_demo.py`, so it won't surprise you.

| # | Example | What to point at |
|---|---|---|
| 1 | Names: Binoche vs Hopkins | Naive Bayes flips (90% → 10%); every other model holds. Only a name changed. |
| 2 | Names again: Kidman vs Seagal | RoBERTa-large, 5× bigger and trained on 15 datasets, **also flips**. Scale doesn't cure it. |
| 3 | Typos: brilliant vs brililant | Our DistilBERT: 100% → 1%. One typo. RoBERTa-large shrugs it off. |
| 4 | Control: a typo in the noun | Nothing moves. It's about losing the *sentiment* word, not any odd word. |
| 5 | Negation: bad vs not bad at all | Word-counting models stay negative; transformers flip. |
| 6 | Sarcasm | Planned failure (below). |

Optional extras: "not boring at all" (shows the low-confidence flag on our DistilBERT), "but I
did" (contrast), double negation, "mundane" and "rollercoaster of meh" (vocabulary).

## Reading the screen
- **Change table (pairs):** a red row means the verdict flipped. A flip on a change that
  shouldn't matter is a shortcut.
- **Word importance:** blue words pushed the verdict positive, red words pushed it negative
  (leave-one-word-out: remove the word and see how much the probability moves).
- **Grey, dotted words:** that classical model never saw the word in training.
- **⚠️ low confidence:** the top probability is below 0.70. Our early-warning result (H4) says
  this signal really does predict errors.
- **Unusual input:** the sentence is further from the training data than 95% of SST-2 test
  sentences. It is *not* labelled "probably wrong", because H4 showed distance doesn't predict
  errors.

## When a live sentence goes wrong: ready lines
- **Sarcasm:** "No model here was trained to detect sarcasm. The words say positive, so our
  DistilBERT says positive. That's shortcut learning in its purest form: reacting to words, not
  meaning." (Shortfall D4 lists sarcasm as out of scope.)
- **Unknown word (greyed out):** "The model has never seen this word. It's guessing from the rest
  of the sentence."
- **Models disagree:** "This is why we tested eight models: their shortcuts are different."
- **A wrong answer with the low-confidence flag:** "And the warning fired. That's H4 working."
- **A wrong answer without it:** "Confidently wrong. That's exactly why accuracy alone is a
  mirage."

## Fine print, if asked
- Logistic regression's Seagal score is 49.8%, so it just tips into "negative". Lead with Naive
  Bayes and RoBERTa-large for the name example.
- RoBERTa-large (SiEBERT) may have been trained on SST-2 and IMDb, so its *accuracy* isn't a fair
  comparison. The demo sentences are our own, so its *behaviour* here is a fair test.
