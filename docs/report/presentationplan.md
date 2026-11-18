# Presentation plan: Mirage, 15 minutes

**The talk in one sentence:** *two AI models can get the same score and behave completely
differently, and this project measures the difference that the score hides: the Mirage Gap.*

Built from [report.md](report.md). Every number below is in the report and comes from
`results/*.csv`. Timings add up to **13 minutes**, which leaves about 2 minutes for questions the
professor asks along the way; the [question bank](#question-bank) has an answer ready for each one
likely to come up.

**What keeps it interesting:**
- **The audience guesses before you reveal.** Three times, ask the room to predict, then show the
  answer (the twins, the name swap, the typo).
- **A live demo in the middle,** not at the end, when attention dips.
- **Stories, not tables:** a horse that "did maths", a model that judges films by the director's
  name, a model that can't read a typo.
- **At most one chart per idea**, and every chart gets a one-line takeaway in its title.
- **It ends with the real world:** where these failures would cost money or harm people.

---

## The run of show

| # | time | slide / action | what it shows |
|---|---|---|---|
| 1 | 0:00 | The hook: "Which one would you hire?" | Two models, same score |
| 2 | 1:00 | Clever Hans | What shortcut learning is |
| 3 | 2:00 | What I built | Data, 8 models, the tests (one slide only) |
| 4 | 3:00 | Looks vs behaves | The idea of the Mirage Gap |
| 5 | 4:00 | **The Mirage Gap chart** | The headline result; reveal the twins |
| 6 | 5:30 | **Live demo** | Names, typo, negation, with the room predicting |
| 7 | 8:30 | Where the gap comes from | The shortcuts are in the data, not the algorithm |
| 8 | 9:45 | Can we see it coming? | Confidence warns, "unusual input" doesn't |
| 9 | 10:30 | Did it hold up? | Pre-registered replication: 16 of 17 |
| 10 | 11:15 | Why this matters outside a classroom | Real-world applications |
| 11 | 12:30 | Take-away | The answer to slide 1 |
| | 13:00 | Questions | |

---

## Slide by slide

### 1. The hook: "Which one would you hire?" (0:00–1:00)

**On screen:** two cards side by side, no model names yet: **Model A: 83.5%** and **Model B:
83.4%** on the standard sentiment test. Underneath: *"Both read movie reviews and say positive or
negative."*

**Say:** "Two AI models took the same exam. One scored 83.5%, the other 83.4%. If you had to deploy
one of them to read your customers' reviews, which would you pick?" *Let the room answer; most will
say "either" or "A". "Hold that thought; by the end of this talk you'll see they are not the same at
all."*

**Why it works:** it starts with a question the audience can answer, and the twins are the
project's strongest result, so the talk is built to pay it off.

### 2. Clever Hans (1:00–2:00)

**On screen:** a picture of a horse and one line: *"Clever Hans could 'count'... until his owner
left the room."*

**Say:** "In the early 1900s a horse in Berlin, Clever Hans, tapped out the answers to arithmetic
questions. He was right almost every time. It turned out he couldn't count at all: he was reading
the tiny reactions of the person asking, and stopped tapping when they relaxed. Right answers,
wrong reason. Machine learning has the same problem, called **shortcut learning**: a model can
score highly by picking up cues that happen to go with the answer, without learning the task. A
sentiment model might learn that *great* means positive, and then call 'not great' positive too."

**Transition:** "So I asked: how much of a model's score is understanding, and how much is Clever
Hans?"

### 3. What I built (2:00–3:00)

**On screen:** one simple diagram, three boxes left to right:
- **Train:** 64,000 movie-review phrases (SST-2);
- **8 models:** five classical models (logistic regression, two Naive Bayes, a decision tree, a
  random forest), our fine-tuned DistilBERT, and two ready-made transformers including RoBERTa-large,
  5× bigger;
- **Test:** the normal test set, 10,000 IMDb reviews from another website, and targeted tests I
  wrote: negation, typos and actors' names.

**Say:** keep it to one sentence per box. "Classical models count words; transformers read words in
context. The targeted tests change *one thing* in a sentence and check whether the model reacts
correctly."

**Don't:** show the accuracy table here. The numbers come on slide 5, with a point.

### 4. Looks vs behaves (3:00–4:00)

**On screen:** two columns.
- **Looks** = accuracy on the normal test set: what a leaderboard shows.
- **Behaves** = robustness: the share of targeted tests the model gets *fully* right. Show the
  three example pairs as text:
  - "The film was good." / "The film was **not** good." → the answer *should* flip.
  - "The movie was brilliant." / "The movie was **brililant**." → it *shouldn't* flip.
  - "Another **Binoche** comedy…" / "Another **Hopkins** comedy…" → it *shouldn't* flip.
- Bottom line, large: **Mirage Gap = Looks − Behaves.**

**Say:** "A pair only counts if the model gets *both* sentences right. That matters: a model that
can't read the sentiment word never changes its answer, so without this rule it would look
perfectly robust." (This is finding 9; one sentence is enough.)

### 5. The Mirage Gap chart (4:00–5:30)

**On screen:** `figures/mirage_gap_slide.png`. Title: **"Accuracy hides how models fail."**

**Say, in this order:**
1. "Each dot is a model. Across: how good it looks. Up: how well it behaves. On the dashed line,
   a model behaves exactly as well as it looks."
2. "**Every model is below the line.** Every one behaves worse than its score suggests."
3. **Reveal the twins:** point at logistic regression and Multinomial NB. "Remember models A and B?
   A is logistic regression, B is Multinomial Naive Bayes. Same score, 83.5% and 83.4%. But B
   behaves **19 points worse**: it has the biggest gap of all, 0.28."
4. "The transformers sit closest to the line. But even they aren't on it."

**Transition:** "Numbers are one thing. Let me show you what a 19-point gap looks like."

### 6. Live demo: the Shortcut Explorer (5:30–8:30)

**On screen:** the app, `http://localhost:8501`, already open, *Show research numbers* on. Use the
curated examples dropdown. **Before each one, ask the room to predict.**

1. **Names: Binoche vs Hopkins (1 min).** "Only the actor's name changes. Will any model change its
   mind?" → Naive Bayes goes from 90% positive to 10%; logistic regression, its twin, doesn't move.
   "Naive Bayes learned that Hopkins appears in negative training phrases, so the name decides the
   verdict. Across 500 name pairs, it flips 38% of the time. The name of an actor should carry no
   sentiment at all."
2. **Typo: brilliant vs brililant (1 min).** "One letter swapped. Will the smart model survive?" →
   Our DistilBERT goes from 100% positive to 1%. RoBERTa-large shrugs it off. "When DistilBERT can't
   read the sentiment word, it defaults to *negative*, because in its training data, broken-up
   unknown words were more common in negative reviews." Then the **control**: a typo in the *noun*
   ("The moive was brilliant.") moves nothing. "So it isn't any typo; it's losing the word that
   carries the sentiment."
3. **Negation: bad vs not bad at all (45 s).** → The transformers flip to positive; logistic
   regression and Naive Bayes stay negative. "The word-counting models see *bad* and stop there.
   This is what reading in context buys."
4. **If time allows:** invite the professor to type a sentence. Have a fallback ready (below).

**If something goes wrong:** the screenshots of every example are in `app/screenshots/`; the ready
lines for a surprising live answer are in [app/PRESENTING.md](../../app/PRESENTING.md). The most
useful one: *"Confidently wrong: that's exactly why accuracy alone is a mirage."*

### 7. Where the gap comes from (8:30–9:45)

**On screen:** three short statements with one icon each, no chart:
- **The shortcuts are in the data.** Three very different algorithms (logistic regression, random
  forest, Naive Bayes) rely on the same words in the same direction: their word effects agree at
  0.88–0.92. Even DistilBERT shares them (0.71–0.77).
- **A pure shortcut:** "IMAX". It appears in 54 training phrases, 96% of them positive. Remove the
  word and DistilBERT becomes noticeably less positive (−0.31). The film's screen format says
  nothing about whether it's good.
- **How a model uses words decides how badly it's fooled.** Naive Bayes lets every word vote at full
  strength, names included; logistic regression shrinks rare words; transformers read context but
  can't read a typo.

**Say:** "The models didn't invent these shortcuts. The training data taught them. That's why
switching algorithms doesn't remove them."

### 8. Can we see it coming? (9:45–10:30)

**On screen:** two bars: **model's own confidence: up to 0.86** vs **"how unusual is this review":
about 0.5 (a coin flip)**, measured as how well each picks out the reviews the model gets wrong.

**Say:** "If models fail quietly, can we at least get a warning? The proposal predicted that
measuring how *unusual* an input is would help. It doesn't: it can tell IMDb from the training data
very well, but not which reviews will be wrong. The model's own confidence does: a low-confidence
answer is a real warning sign. Practical lesson: send low-confidence predictions to a human."

### 9. Did it hold up? (10:30–11:15)

**On screen:** **"16 of 17 claims replicated"**, with one line about the miss.

**Say:** "Several of these analyses were designed after I'd seen the first results, which is a
known trap: if you look long enough, you find patterns by chance. So before touching any new data,
I wrote down 17 claims and committed them. Then I tested them on 10,000 new IMDb reviews and a new
set of test sentences written differently. **Sixteen held.** The one that failed taught me
something: DistilBERT reads 'I would **never** call the movie great' as *positive* 96% of the time.
Its negation skill doesn't cover every way of saying no."

**Why include the failure:** it shows the method is honest, and a professor will trust the other
sixteen more for it.

### 10. Why this matters outside a classroom (11:15–12:30)

**On screen:** four short cases, each with an icon. Pick the two or three that suit the room, and
tell them as stories.

- **Customer reviews and support tickets.** Companies route complaints and track product sentiment
  automatically. Real customers make typos; a model that flips "brilliant" to negative on one typo
  misreads happy customers as angry ones. The typo test is a cheap check before deployment.
- **Names as a hidden bias: hiring and lending.** Our Naive Bayes judged a film by the actor's name.
  The same mechanism in a CV screener or a loan model would judge a *person* by a name, which is
  unfair and in many places illegal. The name-swap test (change only the name, see whether the
  decision changes) is exactly how such bias is audited.
- **Medicine and safety: negation.** In clinical notes, "no evidence of tumour" and "evidence of
  tumour" differ by one word. A model that can't handle negation, which is what our word-counting
  models showed, is dangerous there.
- **Choosing a model.** When a team picks between two models with the same leaderboard score, the
  Mirage Gap is a cheap second number to look at: the twins show the leaderboard alone could pick
  the wrong one.
- **Monitoring after launch.** Confidence works as an alarm and "unusual input" doesn't (slide 8),
  so route low-confidence predictions to people.

**Say, to finish the slide:** "In every one of these, the model would look fine on its test set."

### 11. Take-away (12:30–13:00)

**On screen:** the two cards from slide 1 again, now labelled: **Logistic regression: looks 83.5%,
behaves 74%** and **Multinomial NB: looks 83.4%, behaves 55%**.

**Say:** "So, which would you hire? Now you know they aren't the same. A score tells you how often a
model is right; a few targeted tests tell you *why* it's right, and when it will stop being right.
That difference is the Mirage Gap. Thank you."

---

## Question bank

Answer in one or two sentences, then offer the detail. Each answer's source is linked.

**About the method**
- **"How exactly is robustness computed?"** The mean of four checks: negation pairs, real negated
  sentences, typo pairs and name pairs; a pair passes only if both sentences are right.
  ([report §2.1](report.md#21-the-idea))
- **"Isn't the score arbitrary?"** It is a design choice, defined after the first results. But
  leaving out any one of the four checks keeps the model order the same in 14 of 16 cases, and the
  twins replicated on fresh data. ([report §2.6](report.md#26-how-solid-is-it))
- **"Accuracy and robustness are on different sentences; can you subtract them?"** It's a summary,
  not a probability: compare gaps between models, don't read one gap as "28% of answers are
  wrong for the wrong reason".
- **"Are the differences statistically significant?"** Every comparison has a bootstrap confidence
  interval and Holm correction for multiple tests. The intervals resample whole clusters (all
  sentences with the same adjective, all reviews of the same film), so they aren't too narrow; I
  also require a minimum effect size, not just significance. ([report §5](report.md#5-was-it-tested-fairly))
- **"You wrote the test sentences yourself; isn't that biased?"** Partly, which is why I wrote a
  second set with new wording and re-tested: the conclusions held, but exact numbers depend on the
  wording (finding 15), so I report the conclusions, not the exact template numbers.

**About the models**
- **"Why are logistic regression and Multinomial NB so much more accurate than the trees?"** They
  let every word in the sentence vote; a tree judges a sentence on the few words along one path,
  and Gaussian NB compresses all words into 300 mixed components. ([report §1](report.md#1-the-project-in-one-page))
- **"Why wasn't DistilBERT tuned?"** Standard fine-tuning settings, three seeds; the classical
  models were tuned. That can only understate DistilBERT's advantage, and no conclusion changed
  across its seeds.
- **"Isn't RoBERTa-large's score unfair?"** Possibly: it was probably trained on SST-2 and IMDb
  (it's the only model better on IMDb than on SST-2), so its accuracy is marked as possibly
  inflated. Its behaviour on my own test sentences is a fair test.
- **"Why does DistilBERT treat a typo as negative?"** It splits an unknown word into pieces, and in
  its training data, broken-up words were more common in negative phrases (16.0% vs 10.9% of
  words). A typo in a noun changes nothing. (finding 2)

**About the results**
- **"Which hypotheses held?"** Typos break every model that reads sentiment words (H1); DistilBERT
  is 14–35 points better on real negated sentences (H2); "unusual input" doesn't predict errors
  but confidence does (H4, rejected); the classical models share their shortcut words (H5).
- **"What about H3 and H6?"** Side results: negation marking helps logistic regression; Gaussian
  NB is worse than Multinomial NB (59% vs 83%). Both are in hypothesisanswers.md.
- **"Why does the IMDb accuracy drop?"** Not because reviews are long (length actually helps) but
  because of what reviewers write: plot summaries, mixed opinions. (finding 8)
- **"What failed?"** One pre-registered claim: DistilBERT's negation advantage over the
  single-word models didn't hold on the new sentences, because of the "would never call" blind
  spot.

**About next steps**
- **"How would you fix the shortcuts?"** For the classical models, negation marking already narrows
  the gap. For DistilBERT, the next step is retraining with added negated examples (data
  augmentation) and checking whether the "never" blind spot closes. (report §7)
- **"Does this apply to large language models?"** Untested here; the method (change one thing,
  check the answer) applies directly, and testing newer models is listed as future work.

---

## Backup slides (only if asked)

| if asked about… | show |
|---|---|
| each model's weak spots | `figures/shortcut_fingerprint_slide.png` |
| the hypotheses' numbers | the summary table at the top of [hypothesisanswers.md](../results/hypothesisanswers.md) |
| how the data was split and cleaned | [datasets.md](../data/datasets.md), section 2 and 3 |
| how a model works | the model cards in [docs/models/cards/](../models/cards/) |
| DistilBERT's training | `figures/training_curves.png` |
| negation marking (H3) | `figures/negation_gap_sst2_neg.png` |

## Before the talk

- [ ] **Rehearse twice with a timer.** If you are over 13 minutes, cut slide 8 to one sentence and
      show only two demo examples.
- [ ] **Start the demo before you go in** (`python -m streamlit run app/shortcut_explorer.py`): the
      first start takes 20–40 seconds; after that it works offline. Switch on *Light mode* if the
      laptop is slow.
- [ ] **Open the fallback screenshots** (`app/screenshots/`) in a second window in case the demo
      fails.
- [ ] **Know your three numbers by heart:** 83.5% vs 83.4% (same score), 19 points (the
      behaviour difference), 16 of 17 (replicated).
- [ ] **Find a Clever Hans picture** you are allowed to use (a public-domain photo, with the
      source in small print).
- [ ] Practise the twins reveal on slide 5: it's the moment the talk is built around.
