"""Leakage-safe splitting, feature learning and logistic-regression baselines."""
import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .processing import process_text

LABELS = ["Negative", "Positive"]
VECTORIZERS = ("bow", "bowb", "ng", "tf")


def prepare_data(frame):
    required = {"review", "Sentiment"}
    if not required.issubset(frame.columns):
        raise ValueError("Dataset must contain review and Sentiment columns.")
    data = frame[["review", "Sentiment"]].copy()
    if data.empty:
        raise ValueError("Dataset is empty.")
    if data.isna().any().any():
        raise ValueError("Missing reviews or labels: fix the dataset explicitly.")
    if not data.review.map(lambda v: isinstance(v, str)).all():
        raise ValueError("All reviews must be strings.")
    if not data.Sentiment.isin(LABELS).all():
        raise ValueError("Sentiment labels must be exactly Negative or Positive.")
    data["normalized"] = data.review.map(process_text)
    stats = {"input_rows": len(data), "blank_normalized_rows_removed": int((data.normalized == "").sum())}
    data = data[data.normalized != ""].copy()
    conflicts = data.groupby("normalized").Sentiment.nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    stats["conflicting_text_rows_removed"] = int(data.normalized.isin(conflict_keys).sum())
    data = data[~data.normalized.isin(conflict_keys)].copy()
    stats["duplicate_text_rows_removed"] = int(data.duplicated("normalized").sum())
    data = data.drop_duplicates("normalized").reset_index(drop=True)
    stats["usable_rows"] = len(data)
    stats["label_counts"] = {str(k): int(v) for k, v in data.Sentiment.value_counts().items()}
    if set(data.Sentiment) != set(LABELS) or data.Sentiment.value_counts().min() < 2:
        raise ValueError("At least two distinct usable reviews per class are required.")
    return data, stats


def split_reviews(data, test_size=0.2, seed=42):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    n_test = math.ceil(len(data) * test_size)
    if n_test < 2 or len(data) - n_test < 2:
        raise ValueError("Both train and test sets need room for two classes.")
    train, test = train_test_split(data, test_size=test_size, random_state=seed,
                                   stratify=data.Sentiment)
    if set(train.normalized) & set(test.normalized):
        raise AssertionError("Normalized text overlap across train and test.")
    return train, test


def build_pipeline(vectorizer="tf", min_df=5, seed=42):
    if vectorizer not in VECTORIZERS:
        raise ValueError(f"vectorizer must be one of {VECTORIZERS}")
    if not isinstance(min_df, int) or min_df < 1:
        raise ValueError("min_df must be a positive integer.")
    kwargs = dict(preprocessor=process_text, lowercase=False, min_df=min_df,
                  token_pattern=r"(?u)\b\w+\b")
    # Retain stopwords and negation; no learned preprocessing before splitting.
    if vectorizer == "tf":
        features = TfidfVectorizer(**kwargs)
    else:
        features = CountVectorizer(**kwargs, binary=vectorizer == "bowb",
                                   ngram_range=(1, 3) if vectorizer == "ng" else (1, 1))
    return Pipeline([("features", features),
                     ("classifier", LogisticRegression(max_iter=1000, random_state=seed))])


def vectorize(token_list, y, vect="bow", min_df=5, test_size=0.2, rs=42):
    """Compatibility helper: split text FIRST, fit features on training only.

    Full production path uses prepare_data + split_reviews to remove duplicates.
    """
    train_text, test_text, y_train, y_test = train_test_split(
        token_list, y, test_size=test_size, stratify=y, random_state=rs)
    features = build_pipeline(vect, min_df, rs).named_steps["features"]
    return (features.fit_transform(train_text), features.transform(test_text),
            y_train, y_test, features)
