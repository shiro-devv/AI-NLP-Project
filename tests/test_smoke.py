"""Smoke tests for the analysis pipeline.

Run with: python -m unittest discover -s tests
The sentiment model is skipped here (model=None) so the tests stay fast.
"""

import os
import tempfile
import unittest

from models import analyzer
from utils.report import build_report, write_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "samples")
SAMPLES = [
    os.path.join(SAMPLE_DIR, name) for name in sorted(os.listdir(SAMPLE_DIR))
]


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyzer.analyze_files(SAMPLES, model=None)

    def test_statistics(self):
        for document in self.result["documents"]:
            stats = document["statistics"]
            self.assertGreater(stats["words"], 0)
            self.assertGreater(stats["sentences"], 1)
            self.assertLessEqual(stats["unique_words"], stats["content_words"])

    def test_summary_uses_source_sentences(self):
        for document in self.result["documents"]:
            self.assertTrue(document["summary"]["summary"])
            for sentence in document["summary"]["summary"]:
                self.assertIn(sentence, document["sentences"])
            self.assertLessEqual(len(document["summary"]["summary"]), 7)

    def test_bow_and_tfidf_have_terms(self):
        for document in self.result["documents"]:
            self.assertGreater(document["bag_of_words"]["vocabulary"], 0)
            self.assertTrue(document["bag_of_words"]["terms"])
            self.assertTrue(document["tfidf"]["terms"])
            self.assertTrue(document["phrases"]["bigrams"])

    def test_comparison(self):
        comparison = self.result["comparison"]
        self.assertIsNotNone(comparison)
        self.assertEqual(len(comparison["pairs"]), 3)
        self.assertGreaterEqual(comparison["most_similar"]["score"], 0.0)
        self.assertLessEqual(comparison["most_similar"]["score"], 1.0)

    def test_report_is_saved(self):
        report = build_report(self.result)
        self.assertIn("TEXT STATISTICS", report)
        self.assertIn("SENTIMENT ANALYSIS", report)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "report.txt")
            write_report(report, path)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), report)

    def test_file_limit(self):
        with self.assertRaises(ValueError):
            analyzer.analyze_files(SAMPLES * 2, model=None)


if __name__ == "__main__":
    unittest.main()
