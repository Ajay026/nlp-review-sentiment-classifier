"""Deterministic, corpus-free text normalization shared by train and predict."""
import re
import unicodedata


def process_text(review):
    if not isinstance(review, str):
        raise ValueError("Each review must be a string.")
    text = unicodedata.normalize("NFKC", review).lower().replace("’", "'")
    text = re.sub(r"n['’]t\b", " not", text)
    text = " ".join(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))
    return text
