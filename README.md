# App review sentiment classification

A reproducible classical-NLP portfolio pilot for Ajay Kadiyala: compare four
text representations with logistic regression, without learning the vocabulary
from the test set. Based on a ProjectPro learning project, with a newly written
implementation and explicit changes. This is **not an LLM or GenAI application**.

## What this demonstrates

- Data validation, duplicate handling and privacy-conscious feature selection
- Leakage-safe splitting and train-only vocabulary/IDF learning
- Count, binary-count, n-gram and TF-IDF feature pipelines
- Reproducible evaluation, class-level metrics and inference regression tests

## Quick start

Validated with Python 3.12.14. Use a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For a completely self-contained smoke run with 24 authored synthetic examples:

```bash
python Engine.py --data examples/synthetic_reviews.csv --min-df 1 --output artifacts/synthetic
python predict.py --model artifacts/synthetic/model.joblib --trust-local-model --text "The design editor is useful"
```

This tiny synthetic file only demonstrates the workflow. Do not compare its
scores with the real-data evaluation or claim it is production training data.

On Windows, activate with `.venv\Scripts\activate` instead. Obtain the authorized
local dataset as described in [data/README.md](data/README.md), then run:

```bash
python Engine.py --data /path/to/Canva_reviews.xlsx --vectorizer tf --output artifacts/tf
python predict.py --model artifacts/tf/model.joblib --trust-local-model --text "The design editor is useful and easy to use"
```

Never load downloaded/untrusted pickle or joblib files: they can execute code.
Only use the artifact just produced by your own local training command.
No API key, cloud account, paid service or external model download is needed.

## Repository map

- `Engine.py`: training CLI and aggregate evaluation
- `predict.py`: inference CLI with an explicit trusted-artifact guard
- `Source/model.py`: validation, splitting and pipeline construction
- `Source/processing.py`: shared deterministic text normalization
- `notebooks/01_sentiment_walkthrough.ipynb`: new step-by-step walkthrough
- `tests/`: synthetic offline tests, including leakage regression coverage
- `reports/`: recorded real-data metrics, with no raw reviews or reviewer IDs
- `docs/SOURCE_AND_CHANGES.md`: credit, exact changes and scope boundaries
- `docs/METHODOLOGY.md`: approach and limitations
- `reference/projectpro/`: eight exact source/README/requirements files for comparison only

## Results and interpretation

See [reports/RESULTS.md](reports/RESULTS.md) for the actual local run and the
majority baseline. Do not infer production performance from one holdout.
Models and raw data remain local and are excluded from Git. Eight exact
ProjectPro source/README/requirements files are retained under `reference/` for
comparison, not execution. Original notebooks and the full archive are preserved
privately. The PDF is withheld pending redistribution clarification; the workbook
is withheld to avoid republishing reviewer metadata. See the reference warning.

## Honest portfolio framing

A course-based implementation reviewed and improved with assistant support.
Explain the leakage fix and preprocessing decisions in your own words before
using it in interviews. No fabricated deployment, employer use or business ROI.
This pilot is prepared for review; GitHub publication is a separate verified step.
