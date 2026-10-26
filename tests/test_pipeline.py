import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Engine import train
from Source.model import build_pipeline, prepare_data, split_reviews, vectorize
from Source.processing import process_text
from predict import predict


def fixture():
    # Authored synthetic examples, not copied user reviews.
    return pd.DataFrame({
        'review': [f'useful lovely design editor {chr(97+i)}' for i in range(12)] +
                  [f'broken slow crash editor {chr(97+i)}' for i in range(12)],
        'Sentiment': ['Positive'] * 12 + ['Negative'] * 12})


class SentimentTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(process_text("It ISN’T good! 123"), 'it is not good')
        self.assertEqual(process_text(' café\nDESIGN '), 'café design')

    def test_nonstring_review(self):
        for value in [None, 12, float('nan')]:
            with self.assertRaises(ValueError): process_text(value)

    def test_labels_missing_schema(self):
        for frame in [pd.DataFrame(), fixture().rename(columns={'review': 'text'})]:
            with self.assertRaises(ValueError): prepare_data(frame)

    def test_invalid_labels_and_missing_values(self):
        for value in ['Neutral', 'positive', None]:
            d = fixture(); d.loc[0, 'Sentiment'] = value
            with self.assertRaises(ValueError): prepare_data(d)
        d = fixture(); d.loc[0, 'review'] = None
        with self.assertRaises(ValueError): prepare_data(d)

    def test_duplicates_conflicts_and_blanks(self):
        d = fixture()
        d = pd.concat([d, d.iloc[[0]], pd.DataFrame({'review': [d.iloc[1].review, '!!!'], 'Sentiment': ['Negative', 'Positive']})])
        clean, audit = prepare_data(d)
        self.assertEqual(audit['duplicate_text_rows_removed'], 1)
        self.assertEqual(audit['conflicting_text_rows_removed'], 2)
        self.assertEqual(audit['blank_normalized_rows_removed'], 1)
        self.assertEqual(len(clean), 23)
        a, b = split_reviews(clean)
        self.assertFalse(set(a.normalized) & set(b.normalized))

    def test_single_or_tiny_class(self):
        for frame in [fixture().iloc[:12], fixture().iloc[[0, 12]]]:
            with self.assertRaises(ValueError): prepare_data(frame)

    def test_split_validation_and_reproducibility(self):
        d, _ = prepare_data(fixture())
        for fraction in [0, 1, -0.1, 0.01, 0.99]:
            with self.assertRaises(ValueError): split_reviews(d, fraction)
        a, b = split_reviews(d); c, e = split_reviews(d)
        self.assertEqual(a.index.tolist(), c.index.tolist())
        self.assertEqual(b.index.tolist(), e.index.tolist())

    def test_all_four_modes(self):
        for name in ['bow', 'bowb', 'ng', 'tf']:
            with self.subTest(name=name):
                d, _ = prepare_data(fixture()); a, b = split_reviews(d)
                p = build_pipeline(name, 1).fit(a.review, a.Sentiment)
                self.assertEqual(len(p.predict(b.review)), len(b))
                self.assertEqual(set(p.classes_), {'Negative', 'Positive'})
                self.assertTrue(all(p.predict_proba(b.review).sum(axis=1) > 0.999))

    def test_invalid_config(self):
        for kwargs in [{'vectorizer': 'other'}, {'min_df': 0}, {'min_df': 1.5}]:
            with self.assertRaises(ValueError): build_pipeline(**kwargs)

    def test_pipeline_never_learns_test_only_words(self):
        p = build_pipeline('tf', 1).fit(['nice design', 'awful crash'], ['Positive', 'Negative'])
        vocabulary = dict(p.named_steps['features'].vocabulary_)
        idf = p.named_steps['features'].idf_.copy()
        p.predict(['holdoutsecret nice'])
        self.assertNotIn('holdoutsecret', p.named_steps['features'].vocabulary_)
        self.assertEqual(vocabulary, p.named_steps['features'].vocabulary_)
        np.testing.assert_array_equal(idf, p.named_steps['features'].idf_)

    def test_compatibility_vectorize_splits_before_fitting(self):
        d = fixture(); labels = d.Sentiment.tolist(); texts = d.review.tolist()
        _, test_idx = train_test_split(list(range(len(d))), test_size=0.2, stratify=labels, random_state=42)
        for idx in test_idx: texts[idx] += ' holdoutsecret'
        _, _, _, _, vect = vectorize(texts, labels, vect='tf', min_df=1)
        self.assertNotIn('holdoutsecret', vect.vocabulary_)

    def test_predict_blank_unknown_and_class_mapping(self):
        d = fixture(); p = build_pipeline('tf', 1).fit(d.review, d.Sentiment)
        for text in ['', '!!!', '123']:
            with self.assertRaises(ValueError): predict(p, text)
        r = predict(p, 'zzzzzz'); self.assertEqual(r['known_feature_count'], 0)
        self.assertIsNotNone(r['warning'])
        r = predict(p, 'useful lovely design')
        self.assertAlmostEqual(sum(r['class_probabilities'].values()), 1)
        self.assertEqual(set(r['class_probabilities']), set(p.classes_))

    def test_train_save_load_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root/'test.csv'; fixture().to_csv(source, index=False)
            m = train(source, root/'a', min_df=1)
            n = train(source, root/'b', min_df=1)
            self.assertEqual(m, n)
            self.assertEqual(m['normalized_text_overlap'], 0)
            p = joblib.load(root/'a'/'model.joblib')  # Our own just-created trusted artifact.
            self.assertEqual(predict(p, 'useful design')['label'], 'Positive')
            self.assertEqual(json.loads((root/'a'/'metrics.json').read_text())['test_rows'], 5)

    def test_cli_requires_trust_flag_before_deserialization(self):
        r = subprocess.run([sys.executable, 'predict.py', '--model', 'missing.pkl', '--text', 'hello'], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Refusing to load', r.stderr)

    def test_missing_or_unsupported_data_path(self):
        with self.assertRaises(ValueError): train('does-not-exist.xlsx', '/tmp/not-created')
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'data.txt';p.write_text('test')
            with self.assertRaises(ValueError): train(p, Path(td)/'out')


if __name__ == '__main__':
    unittest.main()
