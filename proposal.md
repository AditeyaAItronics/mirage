# Mirage: Exposing the Illusion of Accuracy

**Detecting Shortcut Learning in Sentiment Classifiers with Classical, Gaussian and Transformer Models**

Project proposal · Aditeya Kayal (2024A8PS0689G) · Solo project, 4 weeks

## Summary

This project investigates whether machine learning models classify the sentiment of text by truly understanding it or by relying on shortcuts such as single keywords. Using the SST-2 movie review dataset, logistic regression on TF-IDF word and word-pair features is used as a baseline and compared against a decision tree, a random forest, Multinomial and Gaussian Naive Bayes, and a fine-tuned DistilBERT transformer. All models are probed with behavioral tests and with real test sentences containing negation. Their decisions are explained using regression weights, tree splits and Integrated Gradients attributions.

I test whether a simple negation-marking trick closes the gap between classical models and the transformer. I also fit a Gaussian model on sentence embeddings to flag unfamiliar inputs, and compare it against the model's own confidence at predicting errors on IMDb reviews. Models are evaluated using accuracy, F1, failure rate per test type, calibration error and error-detection AUROC, to test whether they work for the right reasons.

**Scope:** solo project, 4 weeks, free GPU resources only.

## Motivation and problem statement

High test accuracy does not prove a model has learned the task. A sentiment model can reach over 90% accuracy on a benchmark while relying on spurious cues that happen to correlate with the label in the training data. Common failures include:

- Reading "not good" as positive because the model only notices the word "good".
- Changing a prediction when a single sentiment word is misspelled.
- Labeling "I would love a refund" as positive because of the word "love".

Standard test sets rarely catch these failures, because they contain the same shortcuts as the training data. In industry, they show up as models that break quietly on new customer data, with no warning sign.

The problem this project addresses has three parts:

1. How much of each model's accuracy comes from shortcuts rather than understanding?
2. When is a cheap, interpretable model good enough, and when is a context-aware transformer really needed?
3. Can we tell in advance when a model is likely to be wrong on unfamiliar data?

## Background and related work

**Shortcut learning.** Geirhos et al. (2020) describe shortcut learning as models picking up decision rules that work on standard benchmarks but fail under small, realistic changes. In NLP, McCoy et al. (2019) showed that high-accuracy inference models rely on simple word-overlap heuristics.

**Behavioral testing.** CheckList (Ribeiro et al., 2020) borrows from software testing: it probes models with templated test cases that check simple capabilities, robustness to changes that should not matter, and correct responses to changes that should. It found serious failures that held-out accuracy had missed.

**Classical sentiment classification.** Pang et al. (2002) compared Naive Bayes, maximum entropy and support vector machines for movie review sentiment, and used negation tagging: marking words that follow "not" so that "not good" and "good" become different features. This project revisits that trick to see how much it helps against a modern transformer.

**Explaining predictions.** Linear model weights and decision tree splits can be read directly. For neural models, Integrated Gradients (Sundararajan et al., 2017) attributes a prediction to individual input tokens.

**Detecting likely errors.** Hendrycks and Gimpel (2017) showed that a model's maximum predicted probability is a strong baseline for detecting misclassified and out-of-distribution inputs. Lee et al. (2018) fit Gaussian distributions to a network's internal features and used Mahalanobis distance to flag unfamiliar inputs. Guo et al. (2017) showed that modern neural networks are often overconfident, which motivates measuring calibration.

**Gap this project addresses:** these ideas are usually studied separately and mostly on neural models. This project applies one consistent evaluation to classical and transformer models side by side. It tests a classic fix against a modern baseline, and checks whether a Gaussian early-warning score beats model confidence on real domain shift.

## Research questions

1. **Hidden failures:** How much higher is each model's error rate on negation, typos and other targeted tests than on the normal test set? This is measured both on template tests and on real SST-2 sentences that contain negation (validation and test sentences pooled).
2. **What context buys:** How much better does DistilBERT handle negation than bag-of-words models, and how much of that gap closes when the classical models get word-pair features and simple negation marking?
3. **Early warning:** On IMDb reviews from a different domain, does a Gaussian distance score on sentence embeddings predict a model's errors better than the model's own confidence?
4. **Explanation:** When models fail, which words do they rely on, and do different model families rely on the same words?

**Stretch goal (only if time allows):** fine-tune DistilBERT again with added negated training examples and measure whether negation failures drop.

## Hypotheses

Each hypothesis is tested with a specific comparison. Rejecting any of them is still a reportable finding.

| # | Hypothesis | How it is tested |
| --- | --- | --- |
| H1 | Every model, including DistilBERT, has a clearly higher error rate on negation and sentiment-word typos than on the normal test set | Error rate per test type vs. overall test error |
| H2 | DistilBERT handles negation much better than the bag-of-words models with single-word features | Negation error gap, on templates and on the real negation subset |
| H3 | Word-pair features and negation marking close part, but not all, of that gap | Negation gap before vs. after negation marking |
| H4 | Gaussian distance on sentence embeddings adds information beyond model confidence for predicting errors on IMDb | Error-detection AUROC: distance vs. confidence vs. both combined |
| H5 | Classical models share many of the same top shortcut words, suggesting the shortcuts come from the data | Overlap of top-weighted or top-attributed words across models |
| H6 | Gaussian Naive Bayes performs worse than Multinomial Naive Bayes, because a Gaussian assumption fits text features poorly | Accuracy and F1 of both Naive Bayes variants |

## Datasets

All datasets are public and load directly through Hugging Face Datasets.

| Role | Dataset | Size used | Use in this project |
| --- | --- | --- | --- |
| Training | SST-2 training set (movie review sentences and phrases, binary labels) | 63,971 phrases for training, 3,378 (5%) held out for tuning | Training all models |
| Validation | SST-2 validation set | 872 sentences | Model choices only (e.g. picking DistilBERT's median seed) |
| In-domain test | SST-2 test set, with the original public Stanford labels (from `SetFit/sst2`; GLUE hides its test labels) | 1,820 sentences | Standard accuracy and F1 (the reported in-domain numbers) |
| Real negation subset | SST-2 validation **and** test sentences containing negation words (not, n't, never, no, nothing) | 519 sentences (180 validation + 339 test) | Negation handling on real, not templated, text |
| Out-of-domain test | IMDb movie reviews, random sample | 10,000 reviews | Accuracy drop and the early-warning comparison |
| Behavioral tests | Template-based test cases, written by me | 3,750 cases (6,250 sentences) | Negation, sentiment-word typos, name swaps, name shortcuts |

IMDb reviews are much longer than SST-2 sentences. That difference is part of the domain shift being studied. So that every model reads the same text, each review is capped to its first 220 and last 140 words. DistilBERT reads up to 512 tokens, keeping the start and the end of the review if it is longer, because the verdict is often at the end.

**Leak-free tuning split.** SST-2's training rows are phrases cut from sentences, so a random 5% split would put pieces of the same sentence on both sides. The held-out set is therefore split by whole sentence: a phrase is held out only if every sentence containing it is held out. Test sentences that already appear in training are dropped.

## Models

Six classifiers cover four model families, plus one Gaussian model used only for early warning.

| Model | Family | Input features | Why it is included |
| --- | --- | --- | --- |
| Logistic regression | Regression | TF-IDF, single words and word pairs | Baseline; its weights show shortcut words directly |
| Decision tree | Tree | Same TF-IDF features | Fully readable splits show exactly what drives a decision |
| Random forest | Tree ensemble | Same TF-IDF features | Stronger tree model; feature importances |
| Multinomial Naive Bayes | Probabilistic | Word and word-pair counts | The standard Naive Bayes for text |
| Gaussian Naive Bayes | Gaussian | TF-IDF reduced to ~300 dimensions with truncated SVD | Tests a Gaussian assumption on text features, against Multinomial NB |
| DistilBERT-base | Transformer | Raw text | Context-aware deep model; explained with Integrated Gradients |
| Gaussian density model | Gaussian (not a classifier) | Sentence embeddings (all-MiniLM-L6-v2) | Scores how unfamiliar each input is compared with the training data |

Word-pair (bigram) features let the classical models capture phrases like "not good" as a single feature, which keeps the comparison with DistilBERT fair.

## Methodology

The work runs in five stages, each building on the last.

1. **Train all models.** Fit the classical models with scikit-learn and fine-tune DistilBERT on SST-2. Record standard accuracy, F1 and calibration on the in-domain test set.
2. **Behavioral and negation testing.** Run every model on the template test suite (following the CheckList approach) and on the real negation subset of SST-2. Compare each model's error rate on every test type with its overall test error.
3. **Negation-marking experiment.** Retrain the classical models with negation marking: every word after a negation word, up to the next punctuation mark, gets a NOT_ prefix ("not good movie." becomes "not NOT_good NOT_movie."). A second variant marks only the next three words, with exceptions such as "not only", as a robustness check on the scope rule. Measure how much of the negation gap to DistilBERT closes.
4. **Explanation.** Use each model's own explanation method:
    - Logistic regression: the largest positive and negative word weights.
    - Decision tree: the top splits of a depth-limited tree.
    - Random forest: feature importances.
    - DistilBERT: Integrated Gradients token attributions (Captum library) for failed cases, with a deletion check (remove the top-attributed words and measure how much the prediction changes).

    Then compare the top shortcut words across model families.
5. **Early warning on new data.** Run all classifiers on the IMDb sample. Encode the SST-2 training sentences with a sentence-embedding model, fit a multivariate Gaussian (mean and shrinkage covariance) to the embeddings, and score each IMDb review by its Mahalanobis distance. Each review is scored by its most unusual sentence, because averaging a long review's embedding makes it look typical. Compare three error predictors: the distance score, the model's maximum predicted probability, and a simple logistic combination of both. Review length is scored as a baseline, because the most unusual of many sentences is larger by chance.

## Experimental setup

**Classical models** (scikit-learn, tuned on the 5% held-out split):

| Model | Settings | Tuned |
| --- | --- | --- |
| TF-IDF features | Single words and word pairs, minimum document frequency 2, sublinear term frequency | — |
| Logistic regression | L2 regularization, liblinear solver | C in {0.1, 1, 10, 30, 100} |
| Decision tree | Gini criterion, no depth limit | Minimum leaf size in {10, 20, 50} (very small leaves gave only 0/1 probabilities) |
| Random forest | 300 trees | Max depth in {30, none} |
| Multinomial NB | Count features | Smoothing alpha in {0.1, 0.5, 1} |
| Gaussian NB | Truncated SVD to 300 dimensions | — |

**DistilBERT** (Hugging Face Transformers):

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, learning rate 2e-5, linear decay, 6% warmup |
| Batch size / max length | 32 / 128 tokens (512 for IMDb inference, start + end kept) |
| Epochs | 3, best checkpoint chosen on the held-out split |
| Precision | Mixed (fp16) on a free Kaggle or Colab T4 GPU |
| Seeds | 3 runs (seeds 13, 42, 87); tables report mean ± std, per-sentence analyses use the seed with the median validation accuracy |

**Behavioral test suite** (3,750 cases, 6,250 sentences, generated from templates and word lists):

| Test type | Example | Expected behavior | Cases |
| --- | --- | --- | --- |
| Simple cases | "The movie was superb." | Correct label | 500 |
| Negation | "The movie was not good." | Negative | 500 |
| Negation, direction | "The movie was wonderful." → "The movie was not wonderful." | The model's own answer flips | 500 pairs |
| Typos on sentiment words | "The movie was brilliant." → "The movie was brililant." | Label unchanged | 500 pairs |
| Typo in the noun (control) | "The movie was amazing." → "The moive was amazing." | Label unchanged | 500 pairs |
| Name shortcut | "Another Kidman drama, and it was decent." → "… Seagal …" | Label unchanged | 500 pairs |
| Name swap | "Emily said the movie was brilliant." → "Priya said …" | Label unchanged | 250 pairs |

**Not scored, reported separately:** "not + negative adjective" cases ("The movie was not dreadful.") and their direction pairs (250 + 250). "Not dreadful" isn't clearly positive, so there is no clean expected label.

**Names.** The name-shortcut pairs are film people whose SST-2 training phrases are mostly positive vs mostly negative, chosen from the training labels alone, never from model outputs. They check whether a model judges a review by *who* is mentioned. The name-swap pairs use first names the models never saw in training (Indian vs Western, matched by gender), so they are reported but not counted as robustness: a model can't react to a name it doesn't know.

**Early-warning setup.** The combined error predictor is trained and scored with 5-fold cross-validation on the IMDb sample, so it is never tested on reviews it was fitted on.

**Reproducibility.** Fixed random seeds, config files for every run, results saved to CSV, and all code in a public GitHub repository with a README explaining how to rerun each experiment.

## Evaluation plan and metrics

Evaluation and explanation of results are the core of this project.

| Question | Metric | What it shows |
| --- | --- | --- |
| Standard performance | Accuracy, F1 | The headline number most projects stop at |
| Hidden failures | Error rate per test type and on the real negation subset, relative to overall error | Where models break despite high accuracy |
| What context buys | Negation error gap between DistilBERT and each classical model, before and after negation marking | When a cheap model is good enough |
| Early warning | AUROC at predicting errors: Gaussian distance, model confidence, and both combined | Whether we can tell in advance that a model will be wrong |
| Reliability | Expected calibration error (ECE) and reliability diagrams, on SST-2 and IMDb | Whether model confidence can be trusted, and how it changes under shift |
| Explanation | Overlap of top shortcut words across models; prediction change after deleting top-attributed words | What the models rely on, and whether attributions are faithful |

**Statistical checks.** Bootstrap 95% confidence intervals (1,000 resamples) for every error rate and AUROC. For the real negation subset, which may be small, these intervals show how much weight each result can carry. McNemar's test is used to compare two models' predictions on the same test set. Because many comparisons are made, p-values are Holm-corrected within each family of related tests (for example, all negation gaps).

**How results are reported and explained:**

- A main results table comparing all classifiers on every metric.
- A heatmap of error rates by model and test type.
- A bar chart of the negation gap before and after negation marking.
- A plot of error rate against Gaussian distance on IMDb, with the confidence baseline for comparison.
- Reliability diagrams for the best classical model and DistilBERT.
- A visual of the decision tree's top splits next to DistilBERT attributions for the same sentences.
- 8–10 failure examples, each with a short written explanation of the shortcut involved.

## What counts as a meaningful result

Each research question has an answer that is not known in advance, so every outcome is reportable:

- **Negation marking closes most of the gap:** cheap, interpretable models plus a simple trick can rival a transformer on this weakness. **It doesn't:** the result measures what real context understanding buys.
- **Gaussian distance beats model confidence:** it is a useful early-warning tool for deployed models. **It doesn't:** the model's own confidence is good enough, which is also a practical finding.
- **Different model families rely on the same shortcut words:** the shortcut likely lives in the training data rather than in any one model. **They don't:** each architecture fails in its own way, which argues for testing every model separately.
- **Gaussian NB underperforms Multinomial NB:** the result shows concretely why model assumptions must match the data.

## Timeline

| Week | Focus | Tasks | Checkpoint |
| --- | --- | --- | --- |
| 1 | Train models | Load SST-2 and IMDb, build TF-IDF and SVD features; train 5 classical models and DistilBERT (3 seeds); record accuracy, F1 and calibration | All models trained |
| 2 | Test | Write the behavioral test cases (3,750 in the final suite); run all models on the tests and the real negation subset; retrain classical models with negation marking | Failure and negation results done |
| 3 | Explain and warn | Regression weights, tree splits and forest importances; Integrated Gradients and deletion check for DistilBERT; IMDb run comparing Gaussian distance with model confidence | Explanation and early-warning results done |
| 4 | Report | Bootstrap intervals and McNemar tests; charts, tables and 8–10 explained failures; write report, clean GitHub repo and README | Final submission |

Each week ends with a checkpoint. If one slips, use the fallbacks in the next section rather than squeezing the report week. The stretch goal is attempted in week 3 only if that week's checkpoint is met early.

## Resources, risks and fallbacks

**Compute.** The classical models train in minutes on a laptop CPU. Each DistilBERT run on SST-2 takes well under an hour on a free Kaggle or Colab T4 GPU, so the 3 seeds fit easily within free limits. Google Cloud credits are a backup only.

**Libraries.** scikit-learn, Hugging Face Transformers and Datasets, sentence-transformers, Captum, NumPy, pandas, matplotlib and seaborn.

**Risks and fallbacks:**

| Risk | Fallback |
| --- | --- |
| Short on GPU time | Use 1 seed for DistilBERT instead of 3 |
| Real negation subset is small | Report its size with confidence intervals and rely more on template tests |
| Decision tree too large to read | Limit depth for the explanation figure; report the full tree's accuracy separately |
| Gaussian NB performs very poorly | Report it against Multinomial NB and explain why its assumption fits text badly |
| Integrated Gradients is slow | Run it on failed cases only, about 100 sentences |
| Few failures found | Report which test types models handle well; that is still a finding |
| Falling behind schedule | Drop the stretch goal first, then shrink the number of explained examples; never cut the evaluation tables |

## Expected outcomes and real-world relevance

**Deliverables:**

- A reusable behavioral test suite and evaluation code in a public GitHub repository.
- A report (6–8 pages) answering the four research questions with tables, charts and explained failure examples.
- A README summarizing the key results and how to reproduce them.

**Real-world relevance.** Companies use sentiment models to sort product reviews, support tickets and survey responses. Each part of this project maps to a practical need:

- *Behavioral tests* catch failures like "not worth the money" being read as positive, before a model is deployed.
- *The classical vs. transformer comparison* shows when a cheaper, faster, more explainable model is good enough.
- *The early-warning score* flags unfamiliar inputs, such as reviews in a new language mix like Hinglish, so they can go to a human instead of being guessed.
- *Explanations* show why a model made a decision, which teams and auditors increasingly require.

## Future work: scaling up after the project

The project is designed so each part can grow later into a larger, résumé-level project:

- **More tasks:** add toxicity detection with the Jigsaw Civil Comments dataset and measure identity-term bias (subgroup AUC, BPSN and BNSP metrics).
- **More models:** add RoBERTa, DeBERTa-v3 and a small instruction-tuned LLM evaluated zero-shot, to test whether newer models still take shortcuts.
- **Mitigation:** counterfactual data augmentation for DistilBERT, measured before and after.
- **Multilingual shift:** test on Hindi or Hinglish reviews to make the early-warning experiment more realistic for Indian users.
- **Productization:** a Gradio demo showing a prediction, its token attributions and an "unfamiliar input" warning, plus a reusable testing package other projects can import.

## References

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1).
2. Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*.
3. Guo, C. et al. (2017). On calibration of modern neural networks. *ICML*.
4. Hendrycks, D. and Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*.
5. Lee, K. et al. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *NeurIPS*.
6. Maas, A. et al. (2011). Learning word vectors for sentiment analysis. *ACL* (IMDb dataset).
7. McCoy, T., Pavlick, E. and Linzen, T. (2019). Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. *ACL*.
8. Pang, B., Lee, L. and Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques. *EMNLP*.
9. Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP*.
10. Ribeiro, M. T. et al. (2020). Beyond accuracy: Behavioral testing of NLP models with CheckList. *ACL*.
11. Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.
12. Socher, R. et al. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. *EMNLP* (SST dataset).
13. Sundararajan, M., Taly, A. and Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
