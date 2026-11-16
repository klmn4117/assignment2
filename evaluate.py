"""Validate ten original posts, run a word baseline, and score saved LLM output."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {"positive", "negative"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_posts(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"id", "text", "label", "ambiguous", "rationale"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError("posts.csv: unexpected columns")
        posts = list(reader)
    if len(posts) != 10:
        raise ValueError("Exactly 10 posts are required")
    if any(None in p or any(v is None or not v.strip() for v in p.values()) for p in posts):
        raise ValueError("All cells must be nonempty; extra columns are forbidden")
    if len({p["id"] for p in posts}) != 10 or len({p["text"] for p in posts}) != 10:
        raise ValueError("IDs and post texts must be unique")
    if any(p["label"] not in LABELS or p["ambiguous"] not in {"true", "false"} for p in posts):
        raise ValueError("Invalid label or ambiguous flag")
    if sum(p["ambiguous"] == "true" for p in posts) < 2:
        raise ValueError("At least two ambiguous posts are required")
    return posts


def load_lexicon(path):
    words = read_json(path)
    if not isinstance(words, dict) or set(words) != LABELS:
        raise ValueError("Lexicon must contain positive and negative lists")
    for values in words.values():
        if (not isinstance(values, list) or not values
                or any(not isinstance(w, str) or not w.strip() for w in values)
                or len(set(values)) != len(values)):
            raise ValueError("Word lists must contain unique nonempty strings")
    if set(words["positive"]) & set(words["negative"]):
        raise ValueError("Word lists must not overlap")
    return words


def rule_predict(text, words):
    # Literal substring occurrences; negation and sarcasm are deliberately untreated.
    scores = {label: sum(text.count(w) for w in values) for label, values in words.items()}
    return ("positive" if scores["positive"] > scores["negative"] else "negative"), scores


def score_llm(answer, ids):
    if not isinstance(answer, dict) or set(answer) != {"predictions"}:
        raise ValueError("LLM JSON must contain only predictions")
    rows = answer["predictions"]
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError("LLM output must contain exactly 10 predictions")
    predictions = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "label"}:
            raise ValueError("Each prediction must contain only id and label")
        ident, label = row["id"], row["label"]
        if not isinstance(ident, str) or not isinstance(label, str):
            raise ValueError("IDs and labels must be strings")
        if ident not in ids or ident in predictions or label not in LABELS:
            raise ValueError("Unknown/duplicate ID or invalid label")
        predictions[ident] = label
    return predictions


def validate_metadata(meta, hashes):
    required = {"model", "executed_at", "interface", "settings", "new_chat", "input_sha256"}
    if not isinstance(meta, dict) or not required <= set(meta):
        raise ValueError("Missing LLM execution metadata; see README")
    for key in ("model", "executed_at", "interface"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise ValueError(f"Missing metadata value: {key}")
    if not isinstance(meta["settings"], dict) or meta["new_chat"] is not True:
        raise ValueError("Record settings and use a new chat without gold labels")
    if meta["input_sha256"] != hashes["input_sha256"]:
        raise ValueError("LLM input hash does not match this dataset/prompt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/posts.csv")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon.json")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompts/sentiment.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--llm", type=Path, help="Actual complete JSON response from the LLM")
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    if bool(args.llm) != bool(args.metadata):
        parser.error("--llm and --metadata must be supplied together")
    try:
        posts = load_posts(args.data)
        words = load_lexicon(args.lexicon)
        instruction = args.prompt.read_text(encoding="utf-8-sig").strip()
        if not instruction:
            raise ValueError("Prompt must not be empty")
        llm_input = instruction + "\n" + json.dumps(
            [{"id": p["id"], "text": p["text"]} for p in posts], ensure_ascii=False, indent=2
        ) + "\n"
        hashes = {"data_sha256": digest(args.data), "lexicon_sha256": digest(args.lexicon),
                  "prompt_sha256": digest(args.prompt),
                  "input_sha256": hashlib.sha256(llm_input.encode("utf-8")).hexdigest()}
        predictions = None
        meta = None
        raw = None
        if args.llm:
            predictions = score_llm(read_json(args.llm), {p["id"] for p in posts})
            meta = read_json(args.metadata)
            validate_metadata(meta, hashes)
            raw = args.llm.read_bytes()
        rows = []
        for post in posts:
            label, scores = rule_predict(post["text"], words)
            llm_label = predictions[post["id"]] if predictions is not None else ""
            rows.append({"id": post["id"], "text": post["text"], "gold": post["label"],
                         "ambiguous": post["ambiguous"], "rule": label,
                         "positive_score": scores["positive"], "negative_score": scores["negative"],
                         "rule_correct": label == post["label"], "llm": llm_label,
                         "llm_correct": llm_label == post["label"] if llm_label else ""})
        rule_correct = sum(r["rule_correct"] for r in rows)
        llm_correct = sum(r["llm_correct"] for r in rows) if predictions is not None else None
        summary = {"total": len(rows), **hashes,
                   "rule": {"correct": rule_correct, "accuracy": rule_correct / len(rows),
                            "error_ids": [r["id"] for r in rows if not r["rule_correct"]]},
                   "llm": {"status": "completed" if predictions is not None else "not_run",
                           "correct": llm_correct,
                           "accuracy": llm_correct / len(rows) if llm_correct is not None else None,
                           "error_ids": [r["id"] for r in rows if r["llm_correct"] is False]
                           if predictions is not None else None},
                   "llm_execution": meta}
        # Validate everything before writing. A baseline-only run clears stale LLM scores.
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "llm_input.txt").write_bytes(llm_input.encode("utf-8"))
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (args.out / "predictions.csv").write_bytes(buffer.getvalue().encode("utf-8"))
        (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if raw is not None:
            (args.out / "llm_response.json").write_bytes(raw)
            (args.out / "llm_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
