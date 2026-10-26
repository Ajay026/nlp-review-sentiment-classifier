"""Predict only with a model you trained locally; joblib can execute code."""
import argparse
import json
from pathlib import Path
import joblib
from Source.processing import process_text


def predict(model, text):
    if not process_text(text):
        raise ValueError("Review must contain at least one letter.")
    probabilities = model.predict_proba([text])[0]
    features = model.named_steps["features"].transform([text])
    return {"label": str(model.predict([text])[0]),
            "class_probabilities": {str(c): float(p) for c, p in zip(model.classes_, probabilities)},
            "known_feature_count": int(features.nnz),
            "warning": "No known features; prediction is driven by the intercept." if features.nnz == 0 else None,
            "note": "Model probabilities are not calibrated confidence guarantees."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--trust-local-model", action="store_true", help="Required acknowledgment: load only your own trusted artifact")
    args = parser.parse_args()
    if not args.trust_local_model:
        parser.error("Refusing to load an untrusted artifact. Use --trust-local-model only for your own local training output.")
    if not Path(args.model).is_file():
        parser.error("Model does not exist; train it first.")
    try:
        result = predict(joblib.load(args.model), args.text)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
