# Recorded local evaluation

Run date: 2026-10-25. Python 3.12.14, scikit-learn 1.8.0, pandas 2.2.3.

Input: 1,500 reviews, 1,032 Positive and 468 Negative. No missing review/label, blank normalized text, exact normalized duplicates or conflicting-label duplicate groups were found. Stratified 80/20 split, seed 42: 1,200 train / 300 test. Zero exact normalized-text overlap.

| Representation | Test accuracy | Balanced accuracy | Macro F1 | Vocabulary |
|---|---:|---:|---:|---:|
| bow | 91.67% | 0.9104 | 0.9045 | 475 |
| bowb | 91.33% | 0.9051 | 0.9004 | 475 |
| ng | 91.67% | 0.9133 | 0.9050 | 1157 |
| tf | 94.00% | 0.9245 | 0.9295 | 475 |

Majority-class baseline: accuracy 68.67%, balanced accuracy 0.5000, macro F1 0.4071.

TF-IDF is the predeclared default, not a model chosen by optimizing this test result. The four-mode comparison is exploratory; this is one holdout, not external or production validation.

## Default TF-IDF confusion matrix

Rows are actual, columns predicted; order Negative, Positive.

| Actual / predicted | Negative | Positive |
|---|---:|---:|
| Negative | 83 | 11 |
| Positive | 7 | 199 |

## Reproduce

```bash
python Engine.py --data /path/to/Canva_reviews.xlsx --vectorizer tf --output artifacts/tf
```

Exact per-mode metrics and dataset SHA-256 are in the adjacent JSON files. No row-level reviews, names or profile URLs are recorded. The raw input and model artifacts are deliberately excluded from publication.

## Verification boundaries

- 15 synthetic unit/integration tests passed, including train-only vocabulary/IDF, duplicate/conflict policy, save/load, CLI guard, empty/OOV inputs and reproducibility.
- Four real-data training CLIs and two inference CLI smoke tests completed. No supplied pickle was loaded.
- New notebook code cells ran sequentially under Python; no Jupyter UI/kernel execution was performed.
- Fresh dependency installation on a separate machine, CI on GitHub, deployment and cross-platform operation have not been tested.
- No third-party source or dataset license is inferred from these metrics.
