"""Main orchestrator for Schema-of-Thought experiments."""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.inference import run_inference


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for experiments.
    Orchestrates inference based on configuration.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print(f"Starting experiment: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method.name}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)

    # Apply mode-specific overrides
    if cfg.mode == "sanity":
        print("Applying sanity mode overrides...")
        cfg.run.inference.max_samples = 10
        cfg.wandb.project = f"{cfg.wandb.project}-sanity"
        print(f"  - max_samples: {cfg.run.inference.max_samples}")
        print(f"  - wandb.project: {cfg.wandb.project}")

    elif cfg.mode == "pilot":
        print("Applying pilot mode overrides...")
        cfg.run.inference.max_samples = 100
        cfg.wandb.project = f"{cfg.wandb.project}-pilot"
        print(f"  - max_samples: {cfg.run.inference.max_samples}")
        print(f"  - wandb.project: {cfg.wandb.project}")

    elif cfg.mode == "full":
        print("Running in full mode (no overrides)")
        # Ensure max_samples is None for full dataset
        if cfg.run.inference.max_samples is None:
            print("  - Processing full test set")

    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  Run ID: {cfg.run.run_id}")
    print(f"  Method: {cfg.run.method.name} ({cfg.run.method.type})")
    print(f"  Model: {cfg.run.model.name}")
    print(f"  Dataset: {cfg.run.dataset.name}")
    print(f"  Demo shots: {cfg.run.inference.demo_shots}")
    print(f"  Max samples: {cfg.run.inference.max_samples or 'all'}")
    print(f"  Results dir: {cfg.results_dir}")
    print(f"  WandB project: {cfg.wandb.project}")
    print()

    # Create results directory
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run inference (this is an inference-only task)
    print("Starting inference...")
    metrics = run_inference(cfg)

    print("\n" + "=" * 80)
    print(f"Experiment completed: {cfg.run.run_id}")
    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    main()
