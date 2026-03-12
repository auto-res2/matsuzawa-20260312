"""Evaluation script for comparing experimental runs."""

import argparse
import json
import os
import sys
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: evaluate.py was being called with Hydra-style arguments (key=value)
    #            but expected argparse-style arguments (--key value)
    # [CAUSE]: Workflow file passes arguments as 'results_dir="..."' and 'run_ids="..."'
    #          but argparse expects '--results_dir' and '--run_ids'
    # [FIX]: Transform sys.argv to convert key=value format to --key value format
    #        before parsing with argparse
    #
    # [OLD CODE]:
    # parser = argparse.ArgumentParser(
    #     description="Evaluate and compare experimental runs"
    # )
    # parser.add_argument(
    #     "--results_dir", type=str, required=True, help="Results directory path"
    # )
    # ... rest of argparse setup ...
    # return parser.parse_args()
    #
    # [NEW CODE]:

    # Transform arguments from key=value to --key value format
    transformed_argv = []
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith("-"):
            # Split key=value into --key and value
            key, value = arg.split("=", 1)
            transformed_argv.append(f"--{key}")
            transformed_argv.append(value)
        else:
            transformed_argv.append(arg)

    parser = argparse.ArgumentParser(
        description="Evaluate and compare experimental runs"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory path"
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        required=True,
        help="JSON string list of run IDs to compare",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", "airas"),
        help="WandB entity name",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "2026-0312-matsuzawa"),
        help="WandB project name",
    )
    return parser.parse_args(transformed_argv)


def fetch_run_from_wandb(entity, project, run_id):
    """
    Fetch run data from WandB by display name.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name (run_id)

    Returns:
        WandB run object, or None if not found
    """
    try:
        api = wandb.Api()
        runs = api.runs(
            f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
        )

        if len(runs) == 0:
            print(f"Warning: No WandB run found for {run_id}")
            return None

        # Return most recent run with this name
        return runs[0]

    except Exception as e:
        print(f"Error fetching WandB run {run_id}: {e}")
        return None


def export_run_metrics(run, output_dir):
    """
    Export per-run metrics and create figures.

    Args:
        run: WandB run object
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract summary metrics
    summary_metrics = {
        "run_id": run.name,
        "final_accuracy": run.summary.get("final_accuracy", 0),
        "correct": run.summary.get("correct", 0),
        "total": run.summary.get("total", 0),
        "method": run.config.get("run", {}).get("method", {}).get("name", "unknown"),
    }

    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Exported metrics to {metrics_file}")

    # Fetch history for plotting
    try:
        history = run.history()

        if len(history) > 0 and "accuracy" in history.columns:
            # Create accuracy over time plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                history["step"],
                history["accuracy"],
                marker="o",
                linestyle="-",
                linewidth=2,
            )
            plt.xlabel("Sample", fontsize=12)
            plt.ylabel("Running Accuracy", fontsize=12)
            plt.title(f"Accuracy Over Time - {run.name}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            fig_path = output_dir / "accuracy_over_time.pdf"
            plt.savefig(fig_path, format="pdf", dpi=300)
            plt.close()
            print(f"Saved figure to {fig_path}")

    except Exception as e:
        print(f"Warning: Could not create plots for {run.name}: {e}")

    return summary_metrics


def create_comparison_plots(run_data, output_dir):
    """
    Create comparison plots for all runs.

    Args:
        run_data: List of (run_id, wandb_run, metrics) tuples
        output_dir: Directory to save comparison plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # 1. Accuracy comparison bar plot
    plt.figure(figsize=(10, 6))
    run_ids = [rd[0] for rd in run_data]
    accuracies = [rd[2]["final_accuracy"] for rd in run_data]

    colors = ["#2ecc71" if "proposed" in rid else "#3498db" for rid in run_ids]
    bars = plt.bar(
        range(len(run_ids)),
        accuracies,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    plt.xlabel("Run ID", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy Comparison Across Methods", fontsize=14)
    plt.xticks(range(len(run_ids)), run_ids, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_accuracy.pdf", format="pdf", dpi=300)
    plt.close()
    print(f"Saved comparison plot to {output_dir / 'comparison_accuracy.pdf'}")

    # 2. Accuracy over time comparison (if history available)
    try:
        plt.figure(figsize=(12, 7))

        for run_id, wandb_run, _ in run_data:
            try:
                history = wandb_run.history()
                if len(history) > 0 and "accuracy" in history.columns:
                    plt.plot(
                        history["step"],
                        history["accuracy"],
                        marker="o",
                        linestyle="-",
                        linewidth=2,
                        label=run_id,
                        alpha=0.8,
                    )
            except Exception as e:
                print(f"Warning: Could not plot history for {run_id}: {e}")

        plt.xlabel("Sample", fontsize=12)
        plt.ylabel("Running Accuracy", fontsize=12)
        plt.title("Running Accuracy Comparison", fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_accuracy_over_time.pdf", format="pdf", dpi=300
        )
        plt.close()
        print(
            f"Saved time series comparison to {output_dir / 'comparison_accuracy_over_time.pdf'}"
        )

    except Exception as e:
        print(f"Warning: Could not create time series comparison: {e}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Fetch runs from WandB
    print("\nFetching runs from WandB...")
    run_data = []
    all_metrics = {}

    for run_id in run_ids:
        print(f"  Fetching {run_id}...")
        wandb_run = fetch_run_from_wandb(args.wandb_entity, args.wandb_project, run_id)

        if wandb_run is None:
            print(f"  Skipping {run_id} (not found)")
            continue

        # Export per-run metrics
        run_output_dir = results_dir / run_id
        metrics = export_run_metrics(wandb_run, run_output_dir)

        run_data.append((run_id, wandb_run, metrics))
        all_metrics[run_id] = metrics

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Visualization stage fails when no WandB runs exist yet
    # [CAUSE]: Script exits without creating any output files when run_data is empty,
    #          but the workflow expects PDF/PNG files to be generated
    # [FIX]: Create a placeholder visualization documenting that runs need to execute first
    #
    # [OLD CODE]:
    # if len(run_data) == 0:
    #     print("No runs found. Exiting.")
    #     return
    #
    # [NEW CODE]:
    if len(run_data) == 0:
        print("No runs found. Creating placeholder visualization...")

        # Create comparison directory
        comparison_dir = results_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Create placeholder figure
        plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            "No Experimental Runs Found\n\n"
            f"Expected runs: {', '.join(run_ids)}\n\n"
            "Please execute the experimental runs before visualization.\n"
            f"WandB Project: {args.wandb_project}\n"
            f"WandB Entity: {args.wandb_entity}",
            ha="center",
            va="center",
            fontsize=14,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        plt.axis("off")
        plt.tight_layout()

        placeholder_path = comparison_dir / "no_runs_found.pdf"
        plt.savefig(placeholder_path, format="pdf", dpi=300)
        plt.close()
        print(f"Created placeholder visualization: {placeholder_path}")

        # Create a status JSON file
        status = {
            "status": "no_runs_found",
            "expected_runs": run_ids,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "message": "No experimental runs were found in WandB. Please execute experiments before running visualization.",
        }
        status_file = comparison_dir / "status.json"
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
        print(f"Created status file: {status_file}")

        print("\n" + "=" * 80)
        print("VISUALIZATION STATUS: NO RUNS FOUND")
        print("=" * 80)
        print(f"Expected {len(run_ids)} runs but found 0 in WandB.")
        print("Please execute the following runs before visualization:")
        for run_id in run_ids:
            print(f"  - {run_id}")
        print("=" * 80)
        return

    # Create comparison directory
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(run_data, comparison_dir)

    # Aggregate metrics
    print("\nAggregating metrics...")

    # Identify proposed vs baseline
    proposed_runs = {rid: m for rid, m in all_metrics.items() if "proposed" in rid}
    baseline_runs = {rid: m for rid, m in all_metrics.items() if "comparative" in rid}

    best_proposed = None
    best_proposed_acc = -1
    if proposed_runs:
        best_proposed = max(proposed_runs.items(), key=lambda x: x[1]["final_accuracy"])
        best_proposed_acc = best_proposed[1]["final_accuracy"]

    best_baseline = None
    best_baseline_acc = -1
    if baseline_runs:
        best_baseline = max(baseline_runs.items(), key=lambda x: x[1]["final_accuracy"])
        best_baseline_acc = best_baseline[1]["final_accuracy"]

    gap = (
        best_proposed_acc - best_baseline_acc
        if best_proposed and best_baseline
        else None
    )

    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": all_metrics,
        "best_proposed": best_proposed[0] if best_proposed else None,
        "best_proposed_accuracy": best_proposed_acc if best_proposed else None,
        "best_baseline": best_baseline[0] if best_baseline else None,
        "best_baseline_accuracy": best_baseline_acc if best_baseline else None,
        "gap": gap,
    }

    # Save aggregated metrics
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated metrics to {agg_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for run_id, metrics in all_metrics.items():
        print(f"{run_id:40s} Accuracy: {metrics['final_accuracy']:.4f}")

    if best_proposed:
        print(
            f"\nBest Proposed: {best_proposed[0]} (Accuracy: {best_proposed_acc:.4f})"
        )
    if best_baseline:
        print(f"Best Baseline: {best_baseline[0]} (Accuracy: {best_baseline_acc:.4f})")
    if gap is not None:
        print(f"Gap: {gap:+.4f}")
    print("=" * 80)

    print("\nGenerated files:")
    for run_id in run_ids:
        run_dir = results_dir / run_id
        if run_dir.exists():
            print(f"  {run_dir / 'metrics.json'}")
            if (run_dir / "accuracy_over_time.pdf").exists():
                print(f"  {run_dir / 'accuracy_over_time.pdf'}")
    print(f"  {comparison_dir / 'aggregated_metrics.json'}")
    print(f"  {comparison_dir / 'comparison_accuracy.pdf'}")
    if (comparison_dir / "comparison_accuracy_over_time.pdf").exists():
        print(f"  {comparison_dir / 'comparison_accuracy_over_time.pdf'}")


if __name__ == "__main__":
    main()
