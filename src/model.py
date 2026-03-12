"""Prompt construction logic for Schema-of-Thought and Auto-CoT methods."""

import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
import openai


# Schema templates for different reasoning styles
SCHEMA_TEMPLATES = {
    "decompose": (
        "Break this problem into smaller parts and solve step by step, "
        "identifying key quantities and computing them one at a time."
    ),
    "evidence_aggregation": (
        "List all known facts from the problem, then use them to infer "
        "new facts step by step until reaching the answer."
    ),
    "elimination": (
        "If this is a multiple choice problem, eliminate options one by one "
        "by checking which violate the constraints. Otherwise, verify each "
        "candidate solution systematically."
    ),
    "subgoal_verify": (
        "State a subgoal, solve it, verify the result, then proceed to the "
        "next subgoal until the final answer is reached."
    ),
}


def cluster_questions(
    questions: List[str], num_clusters: int, random_state: int = 42
) -> List[int]:
    """
    Cluster questions using TF-IDF and K-means.

    Args:
        questions: List of question texts
        num_clusters: Number of clusters
        random_state: Random seed for reproducibility

    Returns:
        List of cluster assignments (one per question)
    """
    # Vectorize questions using TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    features = vectorizer.fit_transform(questions)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features)

    return cluster_labels.tolist()


def select_representative_samples(samples: List[Dict], num_clusters: int) -> List[Dict]:
    """
    Select representative samples from clusters (Auto-CoT style).

    Args:
        samples: List of sample dicts with 'question' key
        num_clusters: Number of clusters to create

    Returns:
        List of representative samples (one per cluster)
    """
    questions = [s["question"] for s in samples]
    cluster_labels = cluster_questions(questions, num_clusters)

    # Select one sample from each cluster (first occurrence)
    representatives = []
    seen_clusters = set()

    for sample, label in zip(samples, cluster_labels):
        if label not in seen_clusters:
            representatives.append(sample)
            seen_clusters.add(label)
            if len(representatives) == num_clusters:
                break

    return representatives


def generate_rationale_with_schema(
    question: str,
    schema: str,
    client: openai.OpenAI,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 300,
) -> str:
    """
    Generate a rationale for a question using a specific reasoning schema.

    Args:
        question: Question text
        schema: Schema name (e.g., 'decompose', 'evidence_aggregation')
        client: OpenAI client instance
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated rationale text
    """
    schema_instruction = SCHEMA_TEMPLATES.get(schema, SCHEMA_TEMPLATES["decompose"])

    prompt = f"""Question: {question}

{schema_instruction}

Show your reasoning step by step and end with "The answer is [number]."

Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that solves math problems step by step.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()


def score_rationale(
    rationale: str,
    question: str,
    schema: str,
    answer_consistency_weight: float = 0.3,
    structural_progression_weight: float = 0.25,
    schema_fit_weight: float = 0.25,
    brevity_weight: float = 0.2,
) -> float:
    """
    Score a rationale based on multiple criteria.

    Args:
        rationale: Generated rationale text
        question: Original question
        schema: Schema used to generate rationale
        answer_consistency_weight: Weight for answer extractability
        structural_progression_weight: Weight for structural state progression
        schema_fit_weight: Weight for schema-question fit
        brevity_weight: Weight for brevity

    Returns:
        Weighted score (higher is better)
    """
    # Answer consistency: check if answer is extractable
    answer_score = 1.0 if "answer is" in rationale.lower() else 0.3

    # Structural progression: count explicit state transitions
    # Look for step markers, variable assignments, fact statements
    step_markers = len(
        re.findall(r"(?:step|first|then|next|finally)", rationale.lower())
    )
    equations = len(re.findall(r"=", rationale))
    structural_score = min(1.0, (step_markers + equations) / 5.0)

    # Schema fit: heuristic based on question patterns
    schema_score = compute_schema_fit(question, schema)

    # Brevity: penalize very long rationales
    word_count = len(rationale.split())
    brevity_score = (
        max(0.0, 1.0 - (word_count - 100) / 200) if word_count > 100 else 1.0
    )

    # Weighted combination
    total_score = (
        answer_consistency_weight * answer_score
        + structural_progression_weight * structural_score
        + schema_fit_weight * schema_score
        + brevity_weight * brevity_score
    )

    return total_score


def compute_schema_fit(question: str, schema: str) -> float:
    """
    Compute how well a schema fits a question based on simple heuristics.

    Args:
        question: Question text
        schema: Schema name

    Returns:
        Fit score between 0 and 1
    """
    question_lower = question.lower()

    if schema == "decompose":
        # Good for multi-step arithmetic
        if any(
            word in question_lower
            for word in ["total", "altogether", "combined", "sum"]
        ):
            return 1.0
        return 0.5

    elif schema == "evidence_aggregation":
        # Good for problems with multiple facts
        if question_lower.count(".") > 2 or question_lower.count(",") > 2:
            return 1.0
        return 0.5

    elif schema == "elimination":
        # Good for problems with options or constraints
        if any(
            word in question_lower for word in ["which", "what", "choose", "select"]
        ):
            return 0.8
        return 0.3

    elif schema == "subgoal_verify":
        # Good for problems with clear subproblems
        if any(word in question_lower for word in ["first", "then", "after", "before"]):
            return 0.9
        return 0.5

    return 0.5


def construct_prompt_sot(
    demo_samples: List[Dict],
    test_question: str,
    schemas: List[str],
    client: openai.OpenAI,
    scoring_weights: Dict[str, float],
    schema_diversity_penalty: float = 0.5,
    meta_instruction: str = "Choose the most suitable reasoning style for this problem.",
    model: str = "gpt-3.5-turbo",
) -> str:
    """
    Construct Schema-of-Thought prompt with schema-aware demonstration selection.

    Args:
        demo_samples: List of representative demo samples
        test_question: Test question to answer
        schemas: List of schema names to try
        client: OpenAI client for rationale generation
        scoring_weights: Dictionary of scoring weights
        schema_diversity_penalty: Penalty for duplicate schemas
        meta_instruction: Instruction to prepend to prompt
        model: Model name for rationale generation

    Returns:
        Final prompt string
    """
    demonstrations = []
    used_schemas = []

    for sample in demo_samples:
        question = sample["question"]
        gold_answer = sample["numeric_answer"]

        # Generate rationale candidates for each schema
        best_rationale = None
        best_score = -float("inf")
        best_schema = None

        for schema in schemas:
            try:
                rationale = generate_rationale_with_schema(
                    question, schema, client, model=model
                )
                score = score_rationale(rationale, question, schema, **scoring_weights)

                # Apply diversity penalty if schema already used
                if schema in used_schemas:
                    score -= schema_diversity_penalty

                if score > best_score:
                    best_score = score
                    best_rationale = rationale
                    best_schema = schema
            except Exception as e:
                print(
                    f"Warning: Failed to generate rationale with schema {schema}: {e}"
                )
                continue

        if best_rationale:
            demonstrations.append(f"Q: {question}\nA: {best_rationale}\n")
            used_schemas.append(best_schema)

    # Construct final prompt
    prompt = f"{meta_instruction}\n\n"
    prompt += "\n".join(demonstrations)
    prompt += f"\nQ: {test_question}\nA:"

    return prompt


def construct_prompt_autocot(
    demo_samples: List[Dict],
    test_question: str,
    client: openai.OpenAI,
    meta_instruction: str = "Let's think step by step.",
    model: str = "gpt-3.5-turbo",
) -> str:
    """
    Construct Auto-CoT baseline prompt with simple diversity-based selection.

    Args:
        demo_samples: List of representative demo samples
        test_question: Test question to answer
        client: OpenAI client for rationale generation
        meta_instruction: Instruction to prepend
        model: Model name for rationale generation

    Returns:
        Final prompt string
    """
    demonstrations = []

    for sample in demo_samples:
        question = sample["question"]

        try:
            # Generate rationale with simple decompose schema
            rationale = generate_rationale_with_schema(
                question, "decompose", client, model=model
            )
            demonstrations.append(f"Q: {question}\nA: {rationale}\n")
        except Exception as e:
            print(f"Warning: Failed to generate rationale: {e}")
            continue

    # Construct final prompt
    prompt = f"{meta_instruction}\n\n"
    prompt += "\n".join(demonstrations)
    prompt += f"\nQ: {test_question}\nA:"

    return prompt
