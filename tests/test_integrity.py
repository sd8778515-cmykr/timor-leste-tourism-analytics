"""Regression checks for destination labels, evaluation boundaries and counts."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from category_classifier import apply_category_classification
from config import CATEGORIES, SENTIMENTS
from data_collection import collect
from eda import run_eda
from export_powerbi import build_agg_category_sentiment, build_dim_place
from model_evaluation import validation_sample_preserving_labels
from preprocessing import preprocess
from sentiment_analysis import apply_sentiment_analysis
from supervised_models import _make_models, train_and_evaluate


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = collect(write_log=False)
        clean, _ = preprocess(raw, save=False)
        cls.df = apply_sentiment_analysis(apply_category_classification(clean))

    def test_one_category_per_destination_and_manual_labels_preserved(self):
        self.assertEqual(int(self.df.groupby("place_id", observed=True).category.nunique().max()), 1)
        labeled = self.df.category_label.notna()
        self.assertTrue(self.df.loc[labeled, "category"].astype(str).eq(self.df.loc[labeled, "category_label"]).all())
        shuffled = apply_category_classification(self.df.sample(frac=1, random_state=17))
        pd.testing.assert_series_equal(self.df.category.sort_index(), shuffled.category.sort_index())

    def test_counts_and_percentages_agree(self):
        result = run_eda(self.df, make_figures=False)
        self.assertEqual(result["overview"]["total_reviews"], 3188)
        self.assertEqual(result["overview"]["places_without_review"], 206)
        self.assertEqual(sum(result["category_distribution"].values()), 3188)
        aggregate = build_agg_category_sentiment(self.df)
        for row in aggregate[aggregate.sentiment.eq("Positive")].itertuples():
            self.assertAlmostEqual(row.pct_within_category, result["category_positive_pct"][row.category], delta=0.06)
        self.assertEqual(int(build_dim_place(self.df).n_reviews.sum()), 3188)

    def test_cv_excludes_holdout_and_groups_stay_separate(self):
        small_models = {"Naive Bayes": _make_models()["Naive Bayes"]}
        for target, labels in [("category_label", CATEGORIES), ("sentiment_true_proxy", SENTIMENTS)]:
            cv_indices = []

            def checked_cv(model, X, y, **kwargs):
                cv_indices.extend(X.index)
                if kwargs["groups"] is not None:
                    groups = kwargs["groups"]
                    for train, valid in kwargs["cv"].split(X, y, groups):
                        self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[valid]))
                return cross_val_score(model, X, y, **kwargs)

            with patch("supervised_models._make_models", return_value=small_models), patch("supervised_models.cross_val_score", side_effect=checked_cv):
                result = train_and_evaluate(self.df, target, labels, target)
            self.assertEqual(result["status"], "ok")
            self.assertFalse(set(cv_indices) & set(result["_test_index"]))
            self.assertEqual(set(cv_indices), set(result["_train_index"]))
            if target == "category_label":
                self.assertEqual(result["place_overlap_train_test"], 0)
                self.assertIn("Religious", result["untested_classes"])

    def test_manual_validation_is_preserved(self):
        df = self.df.copy()
        df["cluster"] = None
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            path = Path(temp) / "sample.csv"
            sample = validation_sample_preserving_labels(df, path)
            sample.loc[sample.index[0], "sentiment_manual"] = "Negative"
            sample.loc[sample.index[0], "nota_validador"] = "Human annotation retained"
            sample.to_csv(path, index=False)
            refreshed = validation_sample_preserving_labels(df, path)
            self.assertEqual(refreshed.iloc[0].sentiment_manual, "Negative")
            self.assertEqual(refreshed.iloc[0].nota_validador, "Human annotation retained")
            self.assertEqual(sample.review_id.tolist(), refreshed.review_id.tolist())

    def test_read_only_collection_does_not_write_log(self):
        with patch("data_collection.write_collection_log") as log:
            collect(write_log=False)
            log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
