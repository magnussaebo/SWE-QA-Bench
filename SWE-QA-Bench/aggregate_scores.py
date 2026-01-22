#!/usr/bin/env python3
"""Aggregate SWE-QA scores for a single repo.

Usage:
    python aggregate_scores.py datasets/scores/gpt-4.1-mini/mini_swe_agent/django.jsonl

Output:
    Saves statistics to {repo}_score_stats.txt in the same directory
"""
import json
import sys
from pathlib import Path
import statistics

def aggregate_scores(jsonl_file: Path):
    metrics = ["correctness", "completeness", "relevance", "clarity", "reasoning", "total_score"]
    all_scores = {m: [] for m in metrics}

    with open(jsonl_file) as f:
        for line in f:
            record = json.loads(line)
            for m in metrics:
                if m in record:
                    all_scores[m].append(record[m])

    # Build output
    lines = []
    lines.append(f"Scores from {jsonl_file.parent}")
    lines.append(f"Total records: {len(all_scores['total_score'])}")
    lines.append("")
    for m in metrics:
        vals = all_scores[m]
        if vals:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            lines.append(f"{m:<15} mean={mean:.2f}  std={std:.2f}")

    output = "\n".join(lines)

    # Print to console
    print(output)

    # Save to file
    repo_name = jsonl_file.stem
    output_file = jsonl_file.parent / f"{repo_name}_score_stats.txt"
    output_file.write_text(output)
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aggregate_scores.py <scores_jsonl_file>")
        print("Example: python aggregate_scores.py datasets/scores/gpt-4.1-mini/mini_swe_agent/django.jsonl")
        sys.exit(1)
    aggregate_scores(Path(sys.argv[1]))