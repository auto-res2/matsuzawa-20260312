"""Inference script for Schema-of-Thought and Auto-CoT experiments."""

import os
import json
from pathlib import Path
from tqdm import tqdm
import wandb
import openai
from omegaconf import DictConfig, OmegaConf

from src.preprocess import load_gsm8k, extract_answer_from_response
from src.model import (
    select_representative_samples,
    construct_prompt_sot,
    construct_prompt_autocot,
)


def run_inference(cfg: DictConfig):
    """
    Run inference for a single experiment configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    # Initialize WandB if not disabled
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"WandB run initialized: {wandb.run.url}")

    # Load datasets
    print(f"Loading demo pool from {cfg.run.dataset.demo_split} split...")
    demo_pool = load_gsm8k(
        split=cfg.run.dataset.demo_split,
        cache_dir=".cache",
        max_samples=cfg.run.dataset.max_demo_pool,
    )

    print(f"Loading test set from {cfg.run.dataset.test_split} split...")
    test_samples = load_gsm8k(
        split=cfg.run.dataset.test_split,
        cache_dir=".cache",
        max_samples=cfg.run.inference.max_samples,
    )

    print(f"Demo pool size: {len(demo_pool)}")
    print(f"Test set size: {len(test_samples)}")

    # Select representative demo samples
    print(
        f"Selecting {cfg.run.method.demo_construction.num_clusters} representative demos..."
    )
    demo_samples = select_representative_samples(
        demo_pool, cfg.run.method.demo_construction.num_clusters
    )

    # Limit to specified number of shots
    demo_samples = demo_samples[: cfg.run.inference.demo_shots]
    print(f"Using {len(demo_samples)} demonstration examples")

    # Pre-construct demonstrations (expensive step)
    print("Constructing demonstration prompts...")
    if cfg.run.method.name == "sot_prompt":
        # For SoT, we'll construct prompts per test sample
        # Pre-generate demo rationales here to save time
        demo_rationales = []
        for demo in tqdm(demo_samples, desc="Generating demo rationales"):
            # We'll generate these on-the-fly in construct_prompt_sot
            demo_rationales.append(demo)
        demo_samples_final = demo_rationales
    else:
        # For Auto-CoT baseline, construct once
        demo_rationales = []
        for demo in tqdm(demo_samples, desc="Generating demo rationales"):
            demo_rationales.append(demo)
        demo_samples_final = demo_rationales

    # Run inference on test set
    results = []
    correct = 0
    total = 0

    print(f"\nRunning inference on {len(test_samples)} test samples...")
    for i, test_sample in enumerate(tqdm(test_samples)):
        try:
            # Construct prompt based on method
            if cfg.run.method.name == "sot_prompt":
                prompt = construct_prompt_sot(
                    demo_samples_final,
                    test_sample["question"],
                    cfg.run.method.demo_construction.schemas,
                    client,
                    {
                        "answer_consistency_weight": cfg.run.method.demo_construction.scoring.answer_consistency_weight,
                        "structural_progression_weight": cfg.run.method.demo_construction.scoring.structural_progression_weight,
                        "schema_fit_weight": cfg.run.method.demo_construction.scoring.schema_fit_weight,
                        "brevity_weight": cfg.run.method.demo_construction.scoring.brevity_weight,
                    },
                    cfg.run.method.demo_construction.get(
                        "schema_diversity_penalty", 0.5
                    ),
                    cfg.run.inference.meta_instruction,
                    cfg.run.model.name,
                )
            else:  # auto_cot
                prompt = construct_prompt_autocot(
                    demo_samples_final,
                    test_sample["question"],
                    client,
                    cfg.run.inference.meta_instruction,
                    cfg.run.model.name,
                )

            # Get model response
            response = client.chat.completions.create(
                model=cfg.run.model.name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that solves math problems step by step.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.run.model.temperature,
                max_tokens=cfg.run.model.max_tokens,
            )

            response_text = response.choices[0].message.content.strip()

            # Extract answer
            predicted_answer = extract_answer_from_response(response_text)
            gold_answer = test_sample["numeric_answer"]

            # Check correctness
            is_correct = False
            if predicted_answer and gold_answer:
                try:
                    # Compare as floats with tolerance
                    pred_num = float(predicted_answer)
                    gold_num = float(gold_answer)
                    is_correct = abs(pred_num - gold_num) < 1e-3
                except ValueError:
                    # String comparison as fallback
                    is_correct = predicted_answer == gold_answer

            if is_correct:
                correct += 1
            total += 1

            # Store result
            result = {
                "index": test_sample["index"],
                "question": test_sample["question"],
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "response": response_text,
                "correct": is_correct,
            }
            results.append(result)

            # Log to WandB
            if cfg.wandb.mode != "disabled":
                wandb.log(
                    {
                        "step": i,
                        "correct": int(is_correct),
                        "accuracy": correct / total if total > 0 else 0,
                    }
                )

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            results.append(
                {
                    "index": test_sample.get("index", i),
                    "question": test_sample["question"],
                    "gold_answer": test_sample["numeric_answer"],
                    "predicted_answer": None,
                    "response": None,
                    "correct": False,
                    "error": str(e),
                }
            )
            total += 1

    # Compute final metrics
    accuracy = correct / total if total > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "inference_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "run_id": cfg.run.run_id,
        "method": cfg.run.method.name,
    }

    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    # Log final metrics to WandB
    if cfg.wandb.mode != "disabled":
        wandb.summary["final_accuracy"] = accuracy
        wandb.summary["correct"] = correct
        wandb.summary["total"] = total
        wandb.finish()

    # Sanity/Pilot validation
    if cfg.mode == "sanity":
        validate_sanity(total, accuracy, results)
    elif cfg.mode == "pilot":
        validate_pilot(total, accuracy, results)

    return metrics


def validate_sanity(total_samples, accuracy, results):
    """Validate sanity mode run."""
    # Check minimum samples processed
    if total_samples < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples":{total_samples}, "outputs_valid":false, "outputs_unique":false}}'
        )
        return

    # Check all outputs are valid (not all None)
    valid_outputs = sum(1 for r in results if r.get("predicted_answer") is not None)
    if valid_outputs == 0:
        print(f"SANITY_VALIDATION: FAIL reason=no_valid_outputs")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples":{total_samples}, "outputs_valid":false, "outputs_unique":false}}'
        )
        return

    # Check outputs are not all identical
    answers = [
        r.get("predicted_answer")
        for r in results
        if r.get("predicted_answer") is not None
    ]
    unique_answers = len(set(answers))
    if unique_answers <= 1 and len(answers) > 1:
        print(f"SANITY_VALIDATION: FAIL reason=identical_outputs")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples":{total_samples}, "outputs_valid":true, "outputs_unique":false}}'
        )
        return

    # Check accuracy is finite
    if not (0 <= accuracy <= 1):
        print(f"SANITY_VALIDATION: FAIL reason=invalid_accuracy")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples":{total_samples}, "outputs_valid":true, "outputs_unique":true}}'
        )
        return

    print(f"SANITY_VALIDATION: PASS")
    print(
        f'SANITY_VALIDATION_SUMMARY: {{"samples":{total_samples}, "outputs_valid":true, "outputs_unique":true, "accuracy":{accuracy:.4f}}}'
    )


def validate_pilot(total_samples, accuracy, results):
    """Validate pilot mode run."""
    # Check minimum samples processed
    if total_samples < 50:
        print(f"PILOT_VALIDATION: FAIL reason=insufficient_samples")
        print(
            f'PILOT_VALIDATION_SUMMARY: {{"samples":{total_samples}, "primary_metric":"accuracy", "primary_metric_value":{accuracy:.4f}, "outputs_unique":false}}'
        )
        return

    # Check primary metric is finite
    if not (0 <= accuracy <= 1):
        print(f"PILOT_VALIDATION: FAIL reason=invalid_accuracy")
        print(
            f'PILOT_VALIDATION_SUMMARY: {{"samples":{total_samples}, "primary_metric":"accuracy", "primary_metric_value":{accuracy:.4f}, "outputs_unique":false}}'
        )
        return

    # Check outputs are non-trivial
    answers = [
        r.get("predicted_answer")
        for r in results
        if r.get("predicted_answer") is not None
    ]
    unique_answers = len(set(answers))
    if unique_answers <= 1 and len(answers) > 1:
        print(f"PILOT_VALIDATION: FAIL reason=identical_outputs")
        print(
            f'PILOT_VALIDATION_SUMMARY: {{"samples":{total_samples}, "primary_metric":"accuracy", "primary_metric_value":{accuracy:.4f}, "outputs_unique":false}}'
        )
        return

    print(f"PILOT_VALIDATION: PASS")
    print(
        f'PILOT_VALIDATION_SUMMARY: {{"samples":{total_samples}, "primary_metric":"accuracy", "primary_metric_value":{accuracy:.4f}, "outputs_unique":true}}'
    )


if __name__ == "__main__":
    print("This script should be called from src.main")
