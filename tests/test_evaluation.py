"""Check evaluation conclusions and manual-label handling with controlled data."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import SENTIMENTS
from model_evaluation import evaluate_majority_baseline, evaluate_manual_validation
from evaluation_interpretation import build_interpretation


class EvaluationTests(unittest.TestCase):
    def test_baseline_uses_train_majority_not_test_majority(self):
        df = pd.DataFrame({"target": ["Positive", "Positive", "Negative", "Negative", "Negative"],
                           "partition": ["train", "train", "train", "test", "test"]})
        result = evaluate_majority_baseline(df, "target", "partition", SENTIMENTS, "sentiment")
        self.assertEqual(result["majority_class"], "Positive")
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["n_samples"], 2)

    def test_pending_invalid_and_unmatched_labels_are_not_scored(self):
        df = pd.DataFrame({"review_id": ["a", "b"], "sentiment": ["Positive", "Negative"],
                           "category": ["Modern", "Religious"]})
        sample = pd.DataFrame({"review_id": ["a", "b", "missing"],
                               "sentiment_manual": ["", "INVALID", "Negative"],
                               "category_manual": ["", "", ""]})
        result = evaluate_manual_validation(df, sample)
        self.assertEqual(result["sentiment_completed"], 0)
        self.assertEqual(result["results"]["sentiment"]["n_invalid"], 1)
        self.assertEqual(result["results"]["sentiment"]["n_unmatched"], 1)
        self.assertNotIn("metrics", result["results"]["sentiment"])

    def test_human_comparison_uses_current_predictions(self):
        df = pd.DataFrame({"review_id": ["a", "b", "c"], "sentiment": ["Negative", "Neutral", "Positive"],
                           "category": ["Religious", "Historical", "Modern"]})
        sample = pd.DataFrame({"review_id": ["a", "b", "c"],
                               "sentiment_manual": ["Positive", "Neutral", "Positive"],
                               "category_manual": ["Religious", "Historical", "Modern"],
                               "sentiment": ["Positive", "Neutral", "Positive"]})
        result = evaluate_manual_validation(df, sample)
        self.assertEqual(result["results"]["sentiment"]["status"], "complete")
        self.assertAlmostEqual(result["results"]["sentiment"]["metrics"]["accuracy"], 2/3, places=4)
        self.assertEqual(result["results"]["category"]["metrics"]["accuracy"], 1)

    def test_no_data_interpretation_does_not_claim_success(self):
        result = build_interpretation({"sentiment": {}, "category": {}, "clustering": {}})
        self.assertEqual(result["tasks"]["sentiment"]["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
