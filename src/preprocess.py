"""
Data preprocessing and schema templates for SoT-Prompt experiment.
Provides GSM8K dataset loading and math-focused reasoning schema templates.
"""

import re
from typing import Dict, List, Tuple
from datasets import load_dataset
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def load_gsm8k(split: str = "train", cache_dir: str = ".cache") -> List[Dict]:
    """Load GSM8K dataset from HuggingFace."""
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    examples = []
    for item in dataset:
        question = item["question"]
        answer_text = item["answer"]

        # Extract numeric answer from GSM8K format "#### {number}"
        answer_match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer_text)
        if answer_match:
            numeric_answer = answer_match.group(1).replace(",", "")
        else:
            numeric_answer = None

        examples.append(
            {
                "question": question,
                "answer": numeric_answer,
                "full_answer": answer_text,
            }
        )

    return examples


def get_math_schema_templates() -> Dict[str, Dict[str, str]]:
    """
    Return math-focused schema templates for GSM8K.
    Each schema has a description and a generation prompt template.
    """
    schemas = {
        "decompose": {
            "description": "Break down the problem into smaller arithmetic steps",
            "prompt_template": """Solve this problem by decomposing it into smaller arithmetic steps.
For each step, identify the relevant quantities and perform one calculation.

Question: {question}

Let's solve step by step:""",
        },
        "equation-state-update": {
            "description": "Track quantities as variables and update their state",
            "prompt_template": """Solve this problem by defining variables for key quantities and tracking how they change.
Show explicit state updates: variable = value.

Question: {question}

Let's define variables and track their values:""",
        },
        "subgoal-verify": {
            "description": "State subgoals, solve them, and verify intermediate results",
            "prompt_template": """Solve this problem by breaking it into subgoals.
For each subgoal: state it clearly, solve it, and verify the result makes sense.

Question: {question}

Let's identify subgoals:""",
        },
        "quantity-tracking": {
            "description": "Explicitly track all quantities mentioned in the problem",
            "prompt_template": """Solve this problem by carefully tracking all quantities.
List what you know, what you need to find, and update quantities as you compute.

Question: {question}

Known quantities:
What we need: 
Let's compute:""",
        },
    }

    return schemas


def detect_question_features(question: str) -> Dict[str, float]:
    """
    Extract simple features from a question to estimate schema fit.
    Returns normalized feature scores.
    """
    question_lower = question.lower()

    features = {}

    # Count numbers (indicates arithmetic intensity)
    numbers = re.findall(r"\d+(?:\.\d+)?", question)
    features["num_count"] = len(numbers) / 10.0  # normalize

    # Check for currency/prices (good for quantity tracking)
    features["has_currency"] = (
        1.0
        if any(
            symbol in question for symbol in ["$", "dollars", "cents", "price", "cost"]
        )
        else 0.0
    )

    # Check for rate/ratio words (good for equation state)
    rate_words = ["per", "each", "every", "rate", "times", "multiplied"]
    features["has_rate"] = (
        1.0 if any(word in question_lower for word in rate_words) else 0.0
    )

    # Check for multi-step indicators (good for decompose)
    multistep_words = ["then", "after", "next", "finally", "total", "altogether"]
    features["has_multistep"] = (
        1.0 if any(word in question_lower for word in multistep_words) else 0.0
    )

    # Check for goal-oriented language (good for subgoal)
    goal_words = ["how many", "what is", "find", "calculate", "determine"]
    features["has_goal"] = (
        1.0 if any(phrase in question_lower for phrase in goal_words) else 0.0
    )

    # Question complexity (length proxy)
    features["complexity"] = min(len(question.split()) / 50.0, 1.0)

    return features


def compute_schema_fit(question: str, schema_name: str) -> float:
    """
    Compute a heuristic schema fit score for a question-schema pair.
    Returns a score in [0, 1].
    """
    features = detect_question_features(question)

    # Schema-specific feature weights
    schema_weights = {
        "decompose": {
            "num_count": 0.3,
            "has_multistep": 0.4,
            "complexity": 0.3,
        },
        "equation-state-update": {
            "num_count": 0.2,
            "has_rate": 0.4,
            "has_currency": 0.2,
            "complexity": 0.2,
        },
        "subgoal-verify": {
            "has_multistep": 0.4,
            "has_goal": 0.3,
            "complexity": 0.3,
        },
        "quantity-tracking": {
            "num_count": 0.3,
            "has_currency": 0.3,
            "has_rate": 0.2,
            "has_multistep": 0.2,
        },
    }

    weights = schema_weights.get(schema_name, {})
    score = sum(features.get(feat, 0.0) * weight for feat, weight in weights.items())

    # Normalize to [0, 1]
    return min(max(score, 0.0), 1.0)


def cluster_questions(
    questions: List[str], n_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, List[int]]:
    """
    Cluster questions using K-means on TF-IDF representations.
    Returns cluster labels and representative indices (one per cluster).
    """
    # Compute TF-IDF embeddings
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
    embeddings = vectorizer.fit_transform(questions).toarray()

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Select representative (closest to centroid) from each cluster
    representatives = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id]

        # Find closest point to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = cluster_indices[distances.argmin()]
        representatives.append(int(closest_idx))

    return labels, representatives


def extract_numeric_answer(text: str) -> str | None:
    """
    Extract the final numeric answer from a rationale.
    Looks for patterns like 'answer is X' or '#### X'.
    """
    # Try to find #### marker first (GSM8K format)
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Try common answer patterns
    patterns = [
        r"(?:the )?answer is:?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:final answer|result)(?:\s*is)?:?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).replace(",", "")

    # Last resort: find last number in text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def compute_structure_score(rationale: str) -> float:
    """
    Score a rationale based on structural state progression.
    Rewards explicit variable assignments, numbered steps, and clear transitions.
    Returns score in [0, 1].
    """
    score = 0.0

    # Check for variable assignments (e.g., "x = 5", "total = 10")
    assignments = re.findall(r"\b[a-zA-Z_]\w*\s*=\s*\d+", rationale)
    score += min(len(assignments) / 5.0, 0.3)

    # Check for numbered steps
    numbered_steps = re.findall(r"(?:^|\n)\s*(?:\d+[\.):]|Step\s+\d+)", rationale)
    score += min(len(numbered_steps) / 5.0, 0.3)

    # Check for transition phrases
    transitions = ["then", "next", "after that", "now", "finally", "therefore"]
    transition_count = sum(1 for phrase in transitions if phrase in rationale.lower())
    score += min(transition_count / 5.0, 0.2)

    # Check for intermediate result verification
    verify_phrases = ["which is", "this gives", "so we have", "this means"]
    verify_count = sum(1 for phrase in verify_phrases if phrase in rationale.lower())
    score += min(verify_count / 3.0, 0.2)

    return min(score, 1.0)
