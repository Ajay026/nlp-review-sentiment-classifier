# Methodology and limitations

The task is binary sentiment classification of app reviews. Positive and
Negative are the supplied labels; the labeling procedure is not documented.

1. Read only review and Sentiment; fail on missing/invalid data.
2. Normalize text deterministically to identify blank and repeated texts.
3. Exclude conflicting-label groups and collapse identical normalized text.
4. Stratify the remaining rows into 80% training and 20% test, seed 42.
5. Fit feature vocabulary and IDF (if applicable) on training reviews only.
6. Train logistic regression with a maximum of 1,000 iterations.
7. Evaluate once per fixed configuration against the same held-out partition.
8. Save a newly trained local pipeline, and predict with matching preprocessing.

Four feature modes: word counts (`bow`), binary counts (`bowb`), 1–3-gram counts
(`ng`), and TF-IDF (`tf`). Minimum document frequency is 5 training documents.
TF-IDF is the predeclared default; the four-mode table is exploratory. Selecting
or tuning from that table would require a new untouched test set or nested
cross-validation to support an unbiased final performance claim.

The test partition excludes exact normalized-text overlap. Near duplicates,
same-reviewer correlation and temporal drift can still exist. There is no
external test set, calibration, fairness analysis, deployment/load test or
production monitoring. Accuracy is not a business-impact measurement. Negation,
sarcasm, multilingual text, unseen vocabulary and domain shift remain difficult.

The original ProjectPro PDF explains the instructional progression from
preprocessing to bag-of-words, TF-IDF, logistic regression and prediction.
It is retained privately pending redistribution review; this page is a concise
new explanation of the corrected pilot, not a replacement copy of the PDF.
