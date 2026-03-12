"""
Evaluation script for Schema-of-Thought Prompting experiment.
Fetches results from WandB and generates comparison metrics and figures.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import wandb


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict:
    """
    Fetch run data from WandB API by display name.
    Returns dict with config, summary, and history.
    """
    api = wandb.Api()

    # Find runs with matching display name (most recent)
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        # Try alternate project names (sanity/pilot)
        for suffix in ["-sanity", "-pilot"]:
            runs = api.runs(
                f"{entity}/{project}{suffix}",
                filters={"display_name": run_id},
                order="-created_at",
            )
            if runs:
                break

    if not runs:
        raise ValueError(f"No run found with display_name={run_id}")

    run = runs[0]

    # Extract data
    config = dict(run.config)
    summary = dict(run.summary)

    # Get history (if any time-series metrics exist)
    history = run.history()
    history_dict = history.to_dict("records") if not history.empty else []

    return {
        "run_id": run_id,
        "config": config,
        "summary": summary,
        "history": history_dict,
        "url": run.url,
    }


def export_per_run_metrics(run_data: Dict, output_dir: Path) -> None:
    """Export per-run metrics to JSON."""
    metrics = run_data["summary"]

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Exported metrics: {metrics_path}")


def create_per_run_figures(run_data: Dict, output_dir: Path) -> None:
    """Create per-run visualization figures."""
    summary = run_data["summary"]

    # Figure 1: Accuracy with confidence interval
    fig, ax = plt.subplots(figsize=(6, 4))

    accuracy = summary.get("accuracy", 0)
    accuracy_ci = summary.get("accuracy_ci_95", 0)

    ax.bar(
        ["Accuracy"],
        [accuracy],
        yerr=[accuracy_ci],
        capsize=10,
        color="steelblue",
        alpha=0.7,
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.0])
    ax.set_title(f"Accuracy: {run_data['run_id']}")
    ax.grid(axis="y", alpha=0.3)

    fig_path = output_dir / "accuracy.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Generated figure: {fig_path}")

    # Figure 2: Metrics summary
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Malformed rate
    axes[0].bar(
        ["Malformed\nRate"],
        [summary.get("malformed_rate", 0)],
        color="coral",
        alpha=0.7,
    )
    axes[0].set_ylim([0, 1.0])
    axes[0].set_ylabel("Rate")
    axes[0].grid(axis="y", alpha=0.3)

    # Rationale length
    axes[1].bar(
        ["Avg Rationale\nLength"],
        [summary.get("avg_rationale_length", 0)],
        color="mediumseagreen",
        alpha=0.7,
    )
    axes[1].set_ylabel("Words")
    axes[1].grid(axis="y", alpha=0.3)

    # Schema diversity
    axes[2].bar(
        ["Schema\nDiversity"],
        [summary.get("schema_diversity", 0)],
        color="mediumpurple",
        alpha=0.7,
    )
    axes[2].set_ylim([0, 1.0])
    axes[2].set_ylabel("Diversity Score")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Metrics Summary: {run_data['run_id']}")
    fig_path = output_dir / "metrics_summary.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Generated figure: {fig_path}")


def export_aggregated_metrics(all_run_data: List[Dict], output_dir: Path) -> None:
    """Export aggregated metrics across all runs."""
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": {},
        "best_proposed": None,
        "best_baseline": None,
        "gap": None,
    }

    proposed_runs = []
    baseline_runs = []

    for run_data in all_run_data:
        run_id = run_data["run_id"]
        summary = run_data["summary"]

        metrics = {
            "accuracy": summary.get("accuracy", 0),
            "accuracy_ci_95": summary.get("accuracy_ci_95", 0),
            "malformed_rate": summary.get("malformed_rate", 0),
            "avg_rationale_length": summary.get("avg_rationale_length", 0),
            "schema_diversity": summary.get("schema_diversity", 0),
            "construction_api_calls": summary.get("construction_api_calls", 0),
            "construction_cost_usd": summary.get("construction_cost_usd", 0),
        }

        aggregated["metrics_by_run"][run_id] = metrics

        # Categorize runs
        if "proposed" in run_id:
            proposed_runs.append((run_id, metrics["accuracy"]))
        elif "comparative" in run_id:
            baseline_runs.append((run_id, metrics["accuracy"]))

    # Find best proposed and baseline
    if proposed_runs:
        best_proposed_id, best_proposed_acc = max(proposed_runs, key=lambda x: x[1])
        aggregated["best_proposed"] = {
            "run_id": best_proposed_id,
            "accuracy": best_proposed_acc,
        }

    if baseline_runs:
        best_baseline_id, best_baseline_acc = max(baseline_runs, key=lambda x: x[1])
        aggregated["best_baseline"] = {
            "run_id": best_baseline_id,
            "accuracy": best_baseline_acc,
        }

    # Compute gap
    if aggregated["best_proposed"] and aggregated["best_baseline"]:
        aggregated["gap"] = (
            aggregated["best_proposed"]["accuracy"]
            - aggregated["best_baseline"]["accuracy"]
        )

    # Save
    output_path = output_dir / "aggregated_metrics.json"
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregated metrics saved: {output_path}")

    if aggregated["gap"] is not None:
        print(f"Performance gap (proposed - baseline): {aggregated['gap']:+.4f}")


def create_comparison_figures(all_run_data: List[Dict], output_dir: Path) -> None:
    """Create comparison figures across all runs."""
    run_ids = [rd["run_id"] for rd in all_run_data]

    # Extract metrics
    accuracies = [rd["summary"].get("accuracy", 0) for rd in all_run_data]
    accuracy_cis = [rd["summary"].get("accuracy_ci_95", 0) for rd in all_run_data]
    malformed_rates = [rd["summary"].get("malformed_rate", 0) for rd in all_run_data]
    rationale_lengths = [
        rd["summary"].get("avg_rationale_length", 0) for rd in all_run_data
    ]
    schema_diversities = [
        rd["summary"].get("schema_diversity", 0) for rd in all_run_data
    ]

    # Figure 1: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(run_ids))
    colors = [
        "steelblue"
        if "proposed" in rid
        else "coral"
        if "comparative" in rid
        else "gray"
        for rid in run_ids
    ]

    ax.bar(x, accuracies, yerr=accuracy_cis, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.0])
    ax.set_title("Accuracy Comparison Across Methods")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(["Proposed", "Baseline", "Ablation"], loc="lower right")

    fig_path = output_dir / "comparison_accuracy.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated comparison figure: {fig_path}")

    # Figure 2: Multi-metric comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    axes[0, 0].bar(x, accuracies, color=colors, alpha=0.7)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(run_ids, rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Malformed rate
    axes[0, 1].bar(x, malformed_rates, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(run_ids, rotation=45, ha="right", fontsize=8)
    axes[0, 1].set_ylabel("Malformed Rate")
    axes[0, 1].set_title("Malformed Answer Rate")
    axes[0, 1].grid(axis="y", alpha=0.3)

    # Rationale length
    axes[1, 0].bar(x, rationale_lengths, color=colors, alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(run_ids, rotation=45, ha="right", fontsize=8)
    axes[1, 0].set_ylabel("Avg Words")
    axes[1, 0].set_title("Average Rationale Length")
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Schema diversity
    axes[1, 1].bar(x, schema_diversities, color=colors, alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(run_ids, rotation=45, ha="right", fontsize=8)
    axes[1, 1].set_ylabel("Diversity Score")
    axes[1, 1].set_title("Schema Diversity")
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle("Multi-Metric Comparison Across Methods", fontsize=14)
    fig_path = output_dir / "comparison_multi_metric.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated comparison figure: {fig_path}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate SoT-Prompt experiment results"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON list of run IDs to evaluate"
    )

    args = parser.parse_args()

    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating {len(run_ids)} runs: {run_ids}\n")

    # Get WandB credentials from environment
    wandb_entity = os.environ.get("WANDB_ENTITY", "airas")
    wandb_project = os.environ.get("WANDB_PROJECT", "2026-0312-matsuzawa")

    # Fetch data for each run
    all_run_data = []

    for run_id in run_ids:
        print(f"Fetching data for {run_id}...")
        try:
            run_data = fetch_run_data(wandb_entity, wandb_project, run_id)
            all_run_data.append(run_data)

            # Export per-run metrics and figures
            run_output_dir = Path(args.results_dir) / run_id
            run_output_dir.mkdir(parents=True, exist_ok=True)

            export_per_run_metrics(run_data, run_output_dir)
            create_per_run_figures(run_data, run_output_dir)

            print(f"  WandB URL: {run_data['url']}\n")
        except Exception as e:
            print(f"  Error: {e}\n")
            continue

    if not all_run_data:
        print("No run data fetched. Exiting.")
        return

    # Create comparison outputs
    comparison_dir = Path(args.results_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating comparison outputs...")
    export_aggregated_metrics(all_run_data, comparison_dir)
    create_comparison_figures(all_run_data, comparison_dir)

    print("\nEvaluation complete!")
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
