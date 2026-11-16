import copy
import csv
import tempfile
import unittest
from pathlib import Path
import evaluate


class EvaluationChecks(unittest.TestCase):
    def setUp(self):
        self.posts = evaluate.load_posts(evaluate.ROOT / "data/posts.csv")
        self.ids = {p["id"] for p in self.posts}

    def test_baseline_and_ambiguous_cases(self):
        words = evaluate.load_lexicon(evaluate.ROOT / "data/lexicon.json")
        errors = [p["id"] for p in self.posts
                  if evaluate.rule_predict(p["text"], words)[0] != p["label"]]
        self.assertEqual(errors, ["01", "03", "04", "05", "07"])
        self.assertEqual(evaluate.rule_predict("該当語なし", words)[0], "negative")

    def test_nine_posts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "posts.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.posts[0]))
                writer.writeheader()
                writer.writerows(self.posts[:-1])
            with self.assertRaises(ValueError):
                evaluate.load_posts(path)

    def test_llm_is_scored_by_id_not_row_order(self):
        rows = [{"id": p["id"], "label": p["label"]} for p in reversed(self.posts)]
        self.assertEqual(set(evaluate.score_llm({"predictions": rows}, self.ids)), self.ids)

    def test_bad_llm_responses_are_rejected(self):
        rows = [{"id": p["id"], "label": p["label"]} for p in self.posts]
        variants = [rows[:-1]]
        for field, value in [("id", "01"), ("id", "99"), ("label", "neutral")]:
            bad = copy.deepcopy(rows)
            bad[-1][field] = value
            variants.append(bad)
        for bad in variants:
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                evaluate.score_llm({"predictions": bad}, self.ids)

    def test_metadata_must_match_input(self):
        meta = {"model": "test fixture", "executed_at": "2026-11-16T00:00:00+09:00",
                "interface": "unit test", "settings": {}, "new_chat": True,
                "input_sha256": "old"}
        with self.assertRaises(ValueError):
            evaluate.validate_metadata(meta, {"input_sha256": "new"})


if __name__ == "__main__":
    unittest.main()
