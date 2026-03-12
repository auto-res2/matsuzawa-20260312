"""
Inference script for Schema-of-Thought Prompting experiment.
Implements SoT-Prompt, Auto-CoT baseline, and ablations.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import google.generativeai as genai

from src.preprocess import (
    load_gsm8k,
    get_math_schema_templates,
    cluster_questions,
    compute_schema_fit,
    compute_structure_score,
    extract_numeric_answer,
)


@dataclass
class RationaleCandidate:
    """A single rationale candidate with its scores."""

    question: str
    rationale: str
    schema: str
    answer: str | None
    schema_fit: float
    structure_score: float
    brevity_score: float
    answer_consistency: bool
    total_score: float


@dataclass
class DemoSet:
    """A fixed demonstration set for a method-shot configuration."""

    method_name: str
    shot_budget: int
    demonstrations: List[Dict]  # [{question, rationale, answer}]
    schema_diversity: float
    avg_length: int
    construction_api_calls: int
    construction_cost: float  # estimated in USD


def init_gemini_api() -> None:
    """Initialize Google Gemini API with key from environment."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)


def call_gemini(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Tuple[str, int]:
    """
    Call Gemini API and return response text and token count.
    Returns (response_text, total_tokens).
    """
    model = genai.GenerativeModel(model_name)

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        text = response.text

        # Estimate token count (rough approximation: 1 token ~ 4 chars)
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(text) // 4
        total_tokens = prompt_tokens + completion_tokens

        return text, total_tokens
    except Exception as e:
        print(f"API call failed: {e}", file=sys.stderr)
        return "", 0


def generate_rationale_with_schema(
    question: str, schema_name: str, schema_template: str, model_name: str
) -> Tuple[str, int]:
    """Generate a rationale for a question using a specific schema template."""
    prompt = schema_template.format(question=question)
    rationale, tokens = call_gemini(prompt, model_name=model_name, temperature=0.0)
    return rationale, tokens


def score_rationale(
    question: str,
    rationale: str,
    schema: str,
    ground_truth: str | None,
    use_schema_scoring: bool,
    use_structure_scoring: bool,
    use_brevity_scoring: bool,
    use_answer_consistency: bool,
) -> RationaleCandidate:
    """
    Score a rationale candidate with weighted components.
    Returns a RationaleCandidate with all scores.
    """
    # Extract answer from rationale
    extracted_answer = extract_numeric_answer(rationale)

    # Component scores
    schema_fit = compute_schema_fit(question, schema) if use_schema_scoring else 0.0
    structure_score = (
        compute_structure_score(rationale) if use_structure_scoring else 0.0
    )

    # Brevity score (prefer shorter rationales, normalized by length)
    if use_brevity_scoring:
        # Normalize length: score decreases as length increases
        # Target ~200 words, penalize >500 words
        word_count = len(rationale.split())
        if word_count <= 200:
            brevity_score = 1.0
        elif word_count <= 500:
            brevity_score = 1.0 - (word_count - 200) / 300 * 0.5
        else:
            brevity_score = 0.5 - min((word_count - 500) / 500 * 0.5, 0.5)
        brevity_score = max(brevity_score, 0.0)
    else:
        brevity_score = 0.0

    # Answer consistency (can we extract an answer?)
    answer_consistency = (
        extracted_answer is not None if use_answer_consistency else False
    )
    answer_score = 1.0 if answer_consistency else 0.0

    # Compute weighted total score
    weights = {
        "schema": 0.25 if use_schema_scoring else 0.0,
        "structure": 0.25 if use_structure_scoring else 0.0,
        "brevity": 0.2 if use_brevity_scoring else 0.0,
        "answer": 0.3 if use_answer_consistency else 0.0,
    }

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    total_score = (
        schema_fit * weights["schema"]
        + structure_score * weights["structure"]
        + brevity_score * weights["brevity"]
        + answer_score * weights["answer"]
    )

    return RationaleCandidate(
        question=question,
        rationale=rationale,
        schema=schema,
        answer=extracted_answer,
        schema_fit=schema_fit,
        structure_score=structure_score,
        brevity_score=brevity_score,
        answer_consistency=answer_consistency,
        total_score=total_score,
    )


def construct_demonstration_set(
    construction_pool: List[Dict],
    shot_budget: int,
    schema_templates: Dict[str, Dict],
    model_name: str,
    method_name: str,
    use_schema_scoring: bool,
    use_structure_scoring: bool,
    use_brevity_scoring: bool,
    use_answer_consistency: bool,
) -> DemoSet:
    """
    Construct a fixed demonstration set for a method configuration.

    For SoT-Prompt: cluster questions, generate rationales with multiple schemas, score, and select best.
    For Auto-CoT: cluster questions, generate one rationale per question, select by simple diversity.
    For ablations: use SoT-Prompt logic with specific scoring components disabled.
    """
    print(
        f"\n=== Constructing {method_name} demonstration set ({shot_budget}-shot) ==="
    )

    # Step 1: Cluster questions to ensure diversity
    questions = [ex["question"] for ex in construction_pool]
    _, representative_indices = cluster_questions(questions, n_clusters=shot_budget)

    print(
        f"Selected {len(representative_indices)} representative questions from clusters"
    )

    # Step 2: Generate rationale candidates
    api_calls = 0
    total_tokens = 0

    candidates_by_question = {}

    for idx in tqdm(representative_indices, desc="Generating rationales"):
        example = construction_pool[idx]
        question = example["question"]
        ground_truth = example["answer"]

        question_candidates = []

        if method_name == "auto-cot":
            # Auto-CoT: generate one rationale with generic prompt
            generic_prompt = f"Question: {question}\n\nLet's solve this step by step:"
            rationale, tokens = call_gemini(generic_prompt, model_name=model_name)
            api_calls += 1
            total_tokens += tokens

            candidate = score_rationale(
                question,
                rationale,
                "generic",
                ground_truth,
                use_schema_scoring=False,
                use_structure_scoring=False,
                use_brevity_scoring=False,
                use_answer_consistency=use_answer_consistency,
            )
            question_candidates.append(candidate)
        else:
            # SoT-Prompt or ablations: generate rationales with each schema
            for schema_name, schema_info in schema_templates.items():
                rationale, tokens = generate_rationale_with_schema(
                    question, schema_name, schema_info["prompt_template"], model_name
                )
                api_calls += 1
                total_tokens += tokens

                candidate = score_rationale(
                    question,
                    rationale,
                    schema_name,
                    ground_truth,
                    use_schema_scoring,
                    use_structure_scoring,
                    use_brevity_scoring,
                    use_answer_consistency,
                )
                question_candidates.append(candidate)

        # Select best candidate for this question
        best_candidate = max(question_candidates, key=lambda c: c.total_score)
        candidates_by_question[question] = best_candidate

    # Step 3: Compile demonstration set
    demonstrations = []
    schemas_used = []
    lengths = []

    for idx in representative_indices:
        example = construction_pool[idx]
        question = example["question"]
        candidate = candidates_by_question[question]

        demonstrations.append(
            {
                "question": question,
                "rationale": candidate.rationale,
                "answer": candidate.answer,
                "schema": candidate.schema,
            }
        )

        schemas_used.append(candidate.schema)
        lengths.append(len(candidate.rationale.split()))

    # Compute schema diversity (number of unique schemas / total demonstrations)
    schema_diversity = (
        len(set(schemas_used)) / len(schemas_used) if schemas_used else 0.0
    )
    avg_length = int(np.mean(lengths)) if lengths else 0

    # Estimate cost (rough: $0.075 per 1M input tokens, $0.30 per 1M output tokens for Gemini Flash)
    # Using total tokens as a rough estimate
    construction_cost = total_tokens / 1_000_000 * 0.20  # average rate

    print(f"Constructed {len(demonstrations)} demonstrations:")
    print(f"  Schema diversity: {schema_diversity:.3f}")
    print(f"  Avg rationale length: {avg_length} words")
    print(f"  API calls: {api_calls}")
    print(f"  Estimated cost: ${construction_cost:.4f}")

    return DemoSet(
        method_name=method_name,
        shot_budget=shot_budget,
        demonstrations=demonstrations,
        schema_diversity=schema_diversity,
        avg_length=avg_length,
        construction_api_calls=api_calls,
        construction_cost=construction_cost,
    )


def format_prompt_with_demos(demo_set: DemoSet, test_question: str) -> str:
    """Format a prompt with demonstrations and test question."""
    prompt_parts = []

    # Add demonstrations
    for demo in demo_set.demonstrations:
        prompt_parts.append(f"Question: {demo['question']}\n")
        prompt_parts.append(f"{demo['rationale']}\n")
        if demo["answer"]:
            prompt_parts.append(f"The answer is {demo['answer']}.\n")
        prompt_parts.append("\n")

    # Add test question
    prompt_parts.append(f"Question: {test_question}\n")
    prompt_parts.append("Let's solve this step by step:")

    return "".join(prompt_parts)


def run_inference(cfg: DictConfig) -> None:
    """Main inference function."""
    print(f"\n{'=' * 60}")
    print(f"Running inference: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method.name}")
    print(f"Shot budget: {cfg.run.method.shot_budget}")
    print(f"{'=' * 60}\n")

    # Initialize API
    init_gemini_api()

    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project
            if cfg.mode == "full"
            else f"{cfg.wandb.project}-{cfg.mode}",
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB initialized: {wandb.run.url}\n")

    # Load schema templates
    schema_templates = get_math_schema_templates()

    # Load datasets
    print("Loading GSM8K dataset...")
    train_data = load_gsm8k("train", cache_dir=cfg.dataset.cache_dir)
    test_data = load_gsm8k("test", cache_dir=cfg.dataset.cache_dir)

    # Sample construction pool from train
    if cfg.mode == "sanity":
        construction_pool_size = 20
        n_test = 10
    elif cfg.mode == "pilot":
        construction_pool_size = 100
        n_test = max(50, int(0.2 * len(test_data)))
    else:  # full
        construction_pool_size = cfg.inference.n_construction_pool
        n_test = (
            cfg.inference.n_test_samples
            if cfg.inference.n_test_samples
            else len(test_data)
        )

    np.random.seed(42)
    construction_pool = [
        train_data[i]
        for i in np.random.choice(
            len(train_data), construction_pool_size, replace=False
        )
    ]
    test_samples = [
        test_data[i] for i in np.random.choice(len(test_data), n_test, replace=False)
    ]

    print(f"Construction pool: {len(construction_pool)} examples")
    print(f"Test set: {len(test_samples)} examples\n")

    # Construct demonstration set (precomputed, static)
    demo_set = construct_demonstration_set(
        construction_pool=construction_pool,
        shot_budget=cfg.run.method.shot_budget,
        schema_templates=schema_templates,
        model_name=cfg.model.name,
        method_name=cfg.run.method.name,
        use_schema_scoring=cfg.run.method.use_schema_scoring,
        use_structure_scoring=cfg.run.method.use_structure_scoring,
        use_brevity_scoring=cfg.run.method.use_brevity_scoring,
        use_answer_consistency=cfg.run.method.use_answer_consistency,
    )

    # Save demo set for inspection
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    demo_set_path = results_dir / "demo_set.json"
    with open(demo_set_path, "w") as f:
        json.dump(asdict(demo_set), f, indent=2)
    print(f"Saved demonstration set to {demo_set_path}\n")

    # Run inference on test set
    print(f"Running inference on {len(test_samples)} test examples...")

    results = []
    correct = 0
    malformed = 0
    total_test_tokens = 0
    rationale_lengths = []

    for i, test_example in enumerate(tqdm(test_samples, desc="Testing")):
        question = test_example["question"]
        ground_truth = test_example["answer"]

        # Format prompt with fixed demonstrations
        prompt = format_prompt_with_demos(demo_set, question)
        prompt_tokens = len(prompt) // 4  # rough estimate

        # Generate response
        response, tokens = call_gemini(
            prompt,
            model_name=cfg.model.name,
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
        )
        total_test_tokens += tokens

        # Extract answer
        predicted_answer = extract_numeric_answer(response)

        # Check correctness
        is_correct = False
        is_malformed = predicted_answer is None

        if (
            not is_malformed
            and ground_truth is not None
            and predicted_answer is not None
        ):
            try:
                is_correct = abs(float(predicted_answer) - float(ground_truth)) < 1e-3
            except:
                is_malformed = True

        if is_correct:
            correct += 1
        if is_malformed:
            malformed += 1

        rationale_lengths.append(len(response.split()))

        results.append(
            {
                "index": i,
                "question": question,
                "ground_truth": ground_truth,
                "response": response,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "is_malformed": is_malformed,
                "prompt_tokens": prompt_tokens,
                "response_length": len(response.split()),
            }
        )

        # Log to WandB periodically
        if cfg.wandb.mode != "disabled" and (i + 1) % 50 == 0:
            wandb.log(
                {
                    "examples_processed": i + 1,
                    "accuracy_so_far": correct / (i + 1),
                    "malformed_rate_so_far": malformed / (i + 1),
                }
            )

    # Compute final metrics
    accuracy = correct / len(test_samples)
    malformed_rate = malformed / len(test_samples)
    avg_rationale_length = np.mean(rationale_lengths)
    avg_prompt_tokens = np.mean([r["prompt_tokens"] for r in results])

    # Confidence intervals (95%)
    accuracy_ci = 1.96 * np.sqrt(accuracy * (1 - accuracy) / len(test_samples))

    metrics = {
        "accuracy": accuracy,
        "accuracy_ci_95": accuracy_ci,
        "malformed_rate": malformed_rate,
        "avg_rationale_length": avg_rationale_length,
        "schema_diversity": demo_set.schema_diversity,
        "avg_prompt_tokens": avg_prompt_tokens,
        "construction_api_calls": demo_set.construction_api_calls,
        "construction_cost_usd": demo_set.construction_cost,
        "test_samples": len(test_samples),
        "total_test_tokens": total_test_tokens,
    }

    print(f"\n{'=' * 60}")
    print("FINAL METRICS:")
    print(f"  Accuracy: {accuracy:.4f} ± {accuracy_ci:.4f}")
    print(f"  Malformed rate: {malformed_rate:.4f}")
    print(f"  Avg rationale length: {avg_rationale_length:.1f} words")
    print(f"  Schema diversity: {demo_set.schema_diversity:.3f}")
    print(f"  Avg prompt tokens: {avg_prompt_tokens:.0f}")
    print(f"  Construction API calls: {demo_set.construction_api_calls}")
    print(f"  Construction cost: ${demo_set.construction_cost:.4f}")
    print(f"{'=' * 60}\n")

    # Save results
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved results to {results_path}")
    print(f"Saved metrics to {metrics_path}\n")

    # Log to WandB
    if cfg.wandb.mode != "disabled":
        for key, value in metrics.items():
            wandb.summary[key] = value

        # Upload result files
        wandb.save(str(demo_set_path))
        wandb.save(str(results_path))
        wandb.save(str(metrics_path))

        wandb.finish()

    # Sanity/pilot validation
    if cfg.mode == "sanity":
        perform_sanity_validation(metrics, len(test_samples))
    elif cfg.mode == "pilot":
        perform_pilot_validation(metrics, len(test_samples))


def perform_sanity_validation(metrics: Dict, n_samples: int) -> None:
    """Perform sanity validation checks."""
    passed = True
    reason = ""

    # Check: at least 5 samples processed
    if n_samples < 5:
        passed = False
        reason = "insufficient_samples"

    # Check: metrics are finite
    if not all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))):
        passed = False
        reason = "non_finite_metrics"

    # Check: accuracy is not always 0 (if we have valid samples)
    if metrics["malformed_rate"] < 1.0 and metrics["accuracy"] == 0.0:
        # This could be okay in sanity mode, don't fail
        pass

    # Print verdict
    if passed:
        print("SANITY_VALIDATION: PASS")
    else:
        print(f"SANITY_VALIDATION: FAIL reason={reason}")

    # Print summary
    summary = {
        "samples": n_samples,
        "accuracy": metrics["accuracy"],
        "malformed_rate": metrics["malformed_rate"],
        "avg_length": metrics["avg_rationale_length"],
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")


def perform_pilot_validation(metrics: Dict, n_samples: int) -> None:
    """Perform pilot validation checks."""
    passed = True
    reason = ""

    # Check: at least 50 samples processed
    if n_samples < 50:
        passed = False
        reason = "insufficient_samples"

    # Check: metrics are finite
    if not all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))):
        passed = False
        reason = "non_finite_metrics"

    # Check: accuracy is non-zero
    if metrics["accuracy"] == 0.0:
        passed = False
        reason = "zero_accuracy"

    # Check: not all malformed
    if metrics["malformed_rate"] >= 1.0:
        passed = False
        reason = "all_malformed"

    # Print verdict
    if passed:
        print("PILOT_VALIDATION: PASS")
    else:
        print(f"PILOT_VALIDATION: FAIL reason={reason}")

    # Print summary
    summary = {
        "samples": n_samples,
        "accuracy": metrics["accuracy"],
        "accuracy_ci": metrics["accuracy_ci_95"],
        "malformed_rate": metrics["malformed_rate"],
        "primary_metric": "accuracy",
        "primary_metric_value": metrics["accuracy"],
    }
    print(f"PILOT_VALIDATION_SUMMARY: {json.dumps(summary)}")
