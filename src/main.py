"""
Main orchestration script for Schema-of-Thought Prompting experiment.
Handles mode overrides and invokes inference script.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference import run_inference


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for experiment execution.
    Orchestrates inference runs with mode-specific overrides.
    """
    print(f"Starting experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Results directory: {cfg.results_dir}\n")

    # Apply mode-specific overrides
    if cfg.mode == "sanity":
        print("Applying sanity mode overrides...")
        # Minimal execution for sanity check
        # WandB project override for sanity namespace
        if "sanity" not in cfg.wandb.project:
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"

    elif cfg.mode == "pilot":
        print("Applying pilot mode overrides...")
        # 20% scale for pilot runs
        # WandB project override for pilot namespace
        if "pilot" not in cfg.wandb.project:
            cfg.wandb.project = f"{cfg.wandb.project}-pilot"

    elif cfg.mode == "full":
        print("Running full experiment...")
        # No overrides needed for full mode

    else:
        print(f"Warning: Unknown mode '{cfg.mode}', proceeding with full settings")

    # Print final config
    print("\nFinal configuration:")
    print(OmegaConf.to_yaml(cfg))
    print()

    # Run inference (this is an inference-only experiment)
    try:
        run_inference(cfg)
        print("\nExperiment completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\nExperiment failed with error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
