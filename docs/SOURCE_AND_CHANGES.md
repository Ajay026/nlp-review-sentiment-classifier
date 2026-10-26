# Source and change record

This is a learning portfolio pilot for Ajay Kadiyala, based on the problem taught
in ProjectPro's **NLP Project for Beginners on Text Processing and Classification**,
the first completed project in the **Become a GenAI Engineer** learning path.
It is classical NLP, not a generative model or an LLM project.

Project page: https://www.projectpro.io/project/hackerday-project/project-title/nlp%20text%20processing%20classification%20python

Four authorized downloads were inspected: a modular-code ZIP, a separate model
notebook, a dataset ZIP, and a four-page methodology PDF. Originals and checksums
are preserved privately, not silently overwritten. Original pretrained pickles
were neither executed nor reused. This repository contains a newly written pilot implementation, a new walkthrough
notebook, and an exact reference copy of eight original source/README/requirements
files under `reference/projectpro/`. The reference files are comparison-only, not
the runnable default. No open-source license is asserted for ProjectPro assets.
The official demo explicitly promotes code/dataset GitHub portfolios; original
PDF redistribution remains unresolved and raw-data privacy is treated separately.

## What changed and why

- Split raw review rows before fitting CountVectorizer or TfidfVectorizer. In the
  supplied `Source/model.py` and both notebook copies, fit_transform precedes
  train_test_split. Test reviews can therefore influence vocabulary/IDF.
- Remove exact normalized duplicates and exclude conflicting-label duplicate
  groups before splitting, keeping identical normalized text out of both sets.
  This deterministic rule is fixed in advance, not tuned using test scores.
- Use one sklearn Pipeline so train and inference preprocessing cannot diverge.
- Use deterministic Unicode normalization and regex tokenization. Preserve
  stopwords/negation, expand “n't,” and do not stem. This intentionally differs
  from tutorial NLTK stopword removal and Porter/Lancaster stemming; metrics
  are not a controlled measurement of the leakage fix alone.
- Keep the four instructional vectorizer choices, with TF-IDF fixed as default.
- Report per-class metrics, macro F1, balanced accuracy, confusion matrix and a
  majority baseline, not just accuracy on an imbalanced dataset.
- Add explicit schema/label validation, empty-input errors, unknown-feature
  warnings, deterministic splits and regression tests for train-only fitting.
- Use current installed dependency versions and no NLTK corpus download.
- Exclude raw data, pretrained binaries, reviewer identifiers, caches, secrets
  and original teaching PDFs/notebooks from the publish candidate. Preserve eight
  unmodified source/README/requirements files as labeled comparison references.

The assistant prepared and tested this implementation. Ajay's own review,
understanding and extensions should be described honestly; do not present it as
an independently invented production system or claim business results.
