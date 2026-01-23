#!/usr/bin/env python3
"""Aggregate SWE-QA scores for single or multi-trajectory runs.

Usage:
    # Single file:
    python aggregate_scores.py datasets/scores/gpt-4.1-mini/method/django.jsonl

    # Multi-trajectory (with traj_N subdirs):
    python aggregate_scores.py datasets/scores/gpt_4_1_mini/step_48_multi django.jsonl

Output:
    Single: Saves statistics to {repo}_score_stats.txt in the same directory
    Multi:  Saves per-trajectory stats + overall_django_results.txt in parent dir
"""
import json
import sys
from pathlib import Path
import statistics

METRICS = ["correctness", "completeness", "relevance", "clarity", "reasoning", "total_score"]


def aggregate_single_file(jsonl_file: Path) -> dict:
    """Aggregate scores from a single JSONL file. Returns stats dict."""
    all_scores = {m: [] for m in METRICS}
    empty_answers = 0
    total_records = 0

    with open(jsonl_file) as f:
        for line in f:
            record = json.loads(line)
            total_records += 1

            # Check if answer is empty (agent didn't submit in time)
            # Note: scorer uses "candidate_answer" field
            answer = record.get("candidate_answer", "").strip()
            if not answer:
                empty_answers += 1
                # Count as 0 for all metrics
                for m in METRICS:
                    all_scores[m].append(0)
            else:
                for m in METRICS:
                    if m in record:
                        all_scores[m].append(record[m])

    # Compute stats
    stats = {
        "total_records": total_records,
        "empty_answers": empty_answers,
        "metrics": {}
    }
    for m in METRICS:
        vals = all_scores[m]
        if vals:
            stats["metrics"][m] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0
            }
    return stats


def format_stats(stats: dict, header: str = "") -> str:
    """Format stats dict as text."""
    lines = []
    if header:
        lines.append(header)
    lines.append(f"Total records: {stats['total_records']}")
    lines.append(f"Empty answers (scored as 0): {stats['empty_answers']}")
    lines.append("")
    for m in METRICS:
        if m in stats["metrics"]:
            mean = stats["metrics"][m]["mean"]
            std = stats["metrics"][m]["std"]
            lines.append(f"{m:<15} mean={mean:.2f}  std={std:.2f}")
    return "\n".join(lines)


def aggregate_single(jsonl_file: Path):
    """Original single-file behavior."""
    stats = aggregate_single_file(jsonl_file)
    output = format_stats(stats, f"Scores from {jsonl_file.parent}")

    print(output)

    repo_name = jsonl_file.stem
    output_file = jsonl_file.parent / f"{repo_name}_score_stats.txt"
    output_file.write_text(output)
    print(f"\nSaved to: {output_file}")


def aggregate_multi_trajectory(base_path: Path, target_file: str):
    """Aggregate scores across multiple traj_N subdirectories."""
    # Find all traj_N directories
    traj_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("traj_")])

    if not traj_dirs:
        print(f"No traj_N directories found in {base_path}")
        sys.exit(1)

    repo_name = Path(target_file).stem  # e.g., 'django' from 'django.jsonl'
    all_traj_stats = {}

    print(f"Found {len(traj_dirs)} trajectory directories")
    print(f"Target file: {target_file}")
    print()

    # Process each trajectory
    for traj_dir in traj_dirs:
        traj_name = traj_dir.name
        jsonl_file = traj_dir / target_file

        if not jsonl_file.exists():
            print(f"  {traj_name}: {target_file} not found, skipping")
            continue

        stats = aggregate_single_file(jsonl_file)
        all_traj_stats[traj_name] = stats

        # Save per-trajectory stats
        traj_output = format_stats(stats, f"Scores for {traj_name}")
        traj_stats_file = traj_dir / f"{repo_name}_score_stats.txt"
        traj_stats_file.write_text(traj_output)
        print(f"  {traj_name}: mean total={stats['metrics'].get('total_score', {}).get('mean', 0):.2f}, saved to {traj_stats_file.name}")

    # Build overall summary
    print()
    overall_lines = [
        f"Overall Results for {repo_name}",
        f"Base path: {base_path}",
        f"Trajectories: {len(all_traj_stats)}",
        "=" * 60,
        ""
    ]

    # Per-trajectory summary table
    overall_lines.append("Per-Trajectory Summary:")
    overall_lines.append("-" * 60)
    overall_lines.append(f"{'Trajectory':<12} {'Records':<8} {'Empty':<6} {'Correctness':<12} {'Total Score':<12}")
    overall_lines.append("-" * 60)

    for traj_name in sorted(all_traj_stats.keys()):
        stats = all_traj_stats[traj_name]
        correctness = stats["metrics"].get("correctness", {}).get("mean", 0)
        total = stats["metrics"].get("total_score", {}).get("mean", 0)
        overall_lines.append(
            f"{traj_name:<12} {stats['total_records']:<8} {stats['empty_answers']:<6} {correctness:<12.2f} {total:<12.2f}"
        )

    overall_lines.append("")
    overall_lines.append("")

    # Aggregate across all trajectories
    overall_lines.append("Aggregate Across All Trajectories:")
    overall_lines.append("-" * 60)

    # Combine all scores
    combined = {m: [] for m in METRICS}
    total_records = 0
    total_empty = 0

    for stats in all_traj_stats.values():
        total_records += stats["total_records"]
        total_empty += stats["empty_answers"]
        for m in METRICS:
            if m in stats["metrics"]:
                # Weight by number of records (approximate - use mean * count)
                # Actually, we need the raw values. Let's just average the means for now.
                combined[m].append(stats["metrics"][m]["mean"])

    overall_lines.append(f"Total records across all trajectories: {total_records}")
    overall_lines.append(f"Total empty answers: {total_empty}")
    overall_lines.append("")
    overall_lines.append("Mean of trajectory means:")
    for m in METRICS:
        if combined[m]:
            mean_of_means = statistics.mean(combined[m])
            std_of_means = statistics.stdev(combined[m]) if len(combined[m]) > 1 else 0.0
            overall_lines.append(f"  {m:<15} mean={mean_of_means:.2f}  std={std_of_means:.2f}")

    overall_output = "\n".join(overall_lines)

    # Save overall summary
    overall_file = base_path / f"overall_{repo_name}_results.txt"
    overall_file.write_text(overall_output)
    print(overall_output)
    print()
    print(f"Overall summary saved to: {overall_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:  python aggregate_scores.py <scores_jsonl_file>")
        print("  Multi-traj:   python aggregate_scores.py <base_path> <target_file.jsonl>")
        print()
        print("Examples:")
        print("  python aggregate_scores.py datasets/scores/gpt-4.1-mini/method/django.jsonl")
        print("  python aggregate_scores.py datasets/scores/gpt_4_1_mini/step_48_multi django.jsonl")
        sys.exit(1)

    if len(sys.argv) == 2:
        # Single file mode
        aggregate_single(Path(sys.argv[1]))
    else:
        # Multi-trajectory mode
        aggregate_multi_trajectory(Path(sys.argv[1]), sys.argv[2])