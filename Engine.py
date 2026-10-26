"""Train a fixed baseline and write a model plus aggregate-only evaluation."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from Source.model import LABELS, VECTORIZERS, build_pipeline, prepare_data, split_reviews


def evaluate(y_true, predicted):
    return {"accuracy": float(accuracy_score(y_true, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
            "macro_f1": float(f1_score(y_true, predicted, average="macro")),
            "confusion_matrix_label_order": LABELS,
            "confusion_matrix": confusion_matrix(y_true, predicted, labels=LABELS).tolist(),
            "classification_report": classification_report(y_true, predicted, labels=LABELS,
                                                            output_dict=True, zero_division=0)}


def train(data_path, output_dir, vectorizer="tf", min_df=5, seed=42, test_size=0.2):
    data_path, output_dir = Path(data_path), Path(output_dir)
    if not data_path.is_file():
        raise ValueError(f"Dataset not found: {data_path}. See data/README.md.")
    if data_path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(data_path, usecols=["review", "Sentiment"])
    elif data_path.suffix.lower() == ".csv":
        frame = pd.read_csv(data_path, usecols=["review", "Sentiment"])
    else:
        raise ValueError("Dataset must be .xlsx or .csv.")
    data, audit = prepare_data(frame)
    train_set, test_set = split_reviews(data, test_size, seed)
    model = build_pipeline(vectorizer, min_df, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train_set.review, train_set.Sentiment)
    baseline = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(train_set), 1)), train_set.Sentiment)
    metrics = {"model": "LogisticRegression", "vectorizer": vectorizer,
               "min_df": min_df, "seed": seed, "test_size": test_size,
               "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
               "data_audit": audit, "train_rows": len(train_set), "test_rows": len(test_set),
               "normalized_text_overlap": len(set(train_set.normalized) & set(test_set.normalized)),
               "vocabulary_size": len(model.named_steps["features"].vocabulary_),
               "train": evaluate(train_set.Sentiment, model.predict(train_set.review)),
               "test": evaluate(test_set.Sentiment, model.predict(test_set.review)),
               "majority_baseline_test": evaluate(test_set.Sentiment, baseline.predict(np.zeros((len(test_set), 1)))),
               "runtime": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
               "evaluation_note": "Single stratified holdout, not external validation. Fixed TF-IDF default was chosen before inspecting results. Comparing four modes on this holdout is exploratory, not an unbiased model-selection protocol."}
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Authorized local XLSX/CSV with review and Sentiment columns")
    parser.add_argument("--output", default="artifacts/tf")
    parser.add_argument("--vectorizer", choices=VECTORIZERS, default="tf")
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    try:
        metrics = train(args.data, args.output, args.vectorizer, args.min_df, args.seed, args.test_size)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({k: metrics[k] for k in ["vectorizer", "train_rows", "test_rows", "vocabulary_size", "test"]}, indent=2))


if __name__ == "__main__":
    main()
