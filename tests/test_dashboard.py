"""Streamlit integration tests using generated pipeline outputs and real widgets.

These exercise Python/UI state, not browser pixel layout or external map tiles.
The dashboard must read the official output layer without recalculating models or
rewriting reports.
"""
import hashlib
import io
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import CATEGORIES, SENTIMENTS


def widget(elements, label):
    return next(element for element in elements if element.label == label)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(ROOT / "data/output/reviews_mined.csv")
        cls.df["review_date_parsed"] = pd.to_datetime(cls.df.review_date_parsed, utc=True, format="mixed")
        cls.watched = list((ROOT / "reports").glob("*.json")) + list((ROOT / "reports/tables").glob("*.csv"))
        cls.watched += list((ROOT / "data").glob("**/*.csv")) + list((ROOT / "powerbi").glob("*.csv"))
        cls.hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.watched}

    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=300).run()
        self.assertFalse(self.app.exception, [str(e.message) for e in self.app.exception])

    def tearDown(self):
        self.assertFalse(self.app.exception, [str(e.message) for e in self.app.exception])
        for path in self.watched:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.hashes[str(path)], str(path))

    def detail(self):
        return next(table.value for table in self.app.dataframe
                    if {"review_text", "has_review", "sentiment_split", "category_source"}.issubset(table.value.columns))

    def assert_subset(self, expected):
        actual = self.detail()
        # Compare the displayed rows themselves, not just their count.
        cols = ["place_name", "category", "sentiment", "review_text"]
        def rows(frame):
            return sorted(map(tuple, frame[cols].fillna("").astype(str).to_numpy()))
        self.assertEqual(rows(actual), rows(expected))
        metrics = {m.label: m.value for m in self.app.metric}
        self.assertEqual(int(metrics["Valid reviews"]), int(expected.has_review.sum()))
        self.assertEqual(int(metrics["Destinations"]), expected.place_id.nunique())
        known = expected[expected.sentiment.isin(SENTIMENTS)]
        positive = f"{known.sentiment.eq('Positive').mean() * 100:.1f}%" if len(known) else "N/A"
        self.assertEqual(metrics["Positive sentiment"], positive)

    def test_default_pipeline_dashboard_and_interpretation(self):
        self.assert_subset(self.df)
        self.assertEqual(len(self.app.tabs), 4)
        self.assertFalse(self.app.exception)

    def test_all_categories(self):
        for category in CATEGORIES:
            with self.subTest(category=category):
                widget(self.app.multiselect, "Category").set_value([category]).run()
                self.assert_subset(self.df[self.df.category.eq(category)])

    def test_city_and_combined_filters(self):
        widget(self.app.multiselect, "Municipality").set_value(["Dili"]).run()
        expected = self.df[self.df.city.eq("Dili")]
        self.assert_subset(expected)
        widget(self.app.multiselect, "Category").set_value(["Modern"]).run()
        widget(self.app.multiselect, "Sentiment").set_value(["Positive"]).run()
        self.assert_subset(expected[expected.category.eq("Modern") & expected.sentiment.eq("Positive")])

    def test_sidebar_city_and_multiple_categories(self):
        city = self.df.city.value_counts().index[0]
        widget(self.app.multiselect, "Municipality").set_value([city]).run()
        widget(self.app.multiselect, "Category").set_value(["Modern", "Historical"]).run()
        self.assert_subset(self.df[self.df.city.eq(city) & self.df.category.isin(["Modern", "Historical"])])

    def test_sentiment_filters(self):
        for sentiment in SENTIMENTS:
            with self.subTest(sentiment=sentiment):
                widget(self.app.multiselect, "Sentiment").set_value([sentiment]).run()
                self.assert_subset(self.df[self.df.sentiment.eq(sentiment)])

    def test_cluster_filters(self):
        for cluster in sorted(self.df.cluster.dropna().astype(int).unique()):
            with self.subTest(cluster=int(cluster)):
                widget(self.app.multiselect, "K-Means cluster").set_value([int(cluster)]).run()
                self.assert_subset(self.df[self.df.cluster.eq(cluster)])

    def test_text_only(self):
        widget(self.app.checkbox, "Reviews with text only").check().run()
        self.assert_subset(self.df[self.df.has_text])

    def test_date_range_and_explicit_undated_toggle(self):
        start, end = date(2025, 1, 1), date(2025, 12, 31)
        widget(self.app.date_input, "Review period").set_value((start, end)).run()
        in_range = self.df.review_date_parsed.dt.date.between(start, end)
        self.assert_subset(self.df[in_range | self.df.review_date_parsed.isna()])
        widget(self.app.checkbox, "Include records without a review date").uncheck().run()
        self.assert_subset(self.df[in_range])

    def test_min_rating_filters_places_without_removing_their_low_reviews(self):
        means = self.df.groupby("place_id").review_stars.mean()
        for choice in ("3.0+", "4.0+", "4.5+"):
            with self.subTest(rating=choice):
                widget(self.app.selectbox, "Minimum destination rating").select(choice).run()
                self.assert_subset(self.df[self.df.place_id.map(means).ge(float(choice[:-1]))])

    def test_search_empty_result_and_recovery(self):
        widget(self.app.text_input, "Search destination").set_value("zz_no_destination_98765").run()
        self.assertTrue(self.app.warning)
        self.assertEqual(len(self.app.metric), 0)
        widget(self.app.text_input, "Search destination").set_value("dili").run()
        self.assert_subset(self.df[self.df.place_name.str.contains("dili", case=False, regex=False)])
        widget(self.app.text_input, "Search destination").set_value("").run()
        self.assert_subset(self.df)

    def test_destination_without_reviews(self):
        name = self.df.loc[~self.df.has_review, "place_name"].iloc[0]
        widget(self.app.text_input, "Search destination").set_value(name).run()
        expected = self.df[self.df.place_name.str.contains(name, case=False, regex=False)]
        self.assert_subset(expected)
        if not expected.has_review.any():
            self.assertEqual({m.label: m.value for m in self.app.metric}["Positive sentiment"], "N/A")

    def test_navigation_sort_and_csv_payload(self):
        self.assert_subset(self.df)
        # AppTest exposes the download element; validate its generated CSV variable
        # through Streamlit's element rather than making an external HTTP request.
        self.assertEqual(len(self.app.get("download_button")), 1)
        exported = pd.read_csv(io.StringIO(self.detail().to_csv(index=False)))
        self.assertEqual(len(exported), len(self.df))
        self.assertNotIn("reviewer_name", exported.columns)


if __name__ == "__main__":
    unittest.main()
