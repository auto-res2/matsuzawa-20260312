"""Dataset preprocessing and loading for GSM8K."""

import re
from datasets import load_dataset
from pathlib import Path


def load_gsm8k(split="train", cache_dir=".cache", max_samples=None):
    """
    Load GSM8K dataset from HuggingFace.

    Args:
        split: Dataset split ('train' or 'test')
        cache_dir: Cache directory for downloaded datasets
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=str(cache_path))

    samples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        question = item["question"]
        answer_text = item["answer"]

        # Extract numeric answer from answer text
        # GSM8K answers have format "... #### ANSWER"
        numeric_answer = extract_numeric_answer(answer_text)

        samples.append(
            {
                "question": question,
                "answer": answer_text,
                "numeric_answer": numeric_answer,
                "index": i,
            }
        )

    return samples


def extract_numeric_answer(answer_text):
    """
    Extract the numeric answer from GSM8K answer text.
    GSM8K uses format: "reasoning steps... #### numeric_answer"

    Args:
        answer_text: Full answer text with reasoning

    Returns:
        Numeric answer as string, or None if not found
    """
    # Look for the #### marker
    if "####" in answer_text:
        answer_part = answer_text.split("####")[-1].strip()
        # Remove commas and extract number
        answer_part = answer_part.replace(",", "")
        # Extract first number found
        match = re.search(r"-?\d+\.?\d*", answer_part)
        if match:
            return match.group(0)
    return None


def extract_answer_from_response(response_text):
    """
    Extract numeric answer from model response.
    Looks for common answer patterns in CoT responses.

    Args:
        response_text: Model's generated response

    Returns:
        Extracted numeric answer as string, or None
    """
    # Try to find "The answer is X" pattern
    patterns = [
        r"[Tt]he answer is:?\s*\$?\s*(-?\d+\.?\d*)",
        r"[Aa]nswer:?\s*\$?\s*(-?\d+\.?\d*)",
        r"####\s*(-?\d+\.?\d*)",
        r"=\s*\$?\s*(-?\d+\.?\d*)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1).replace(",", "")

    # Fallback: extract last number in the text
    numbers = re.findall(r"-?\d+\.?\d*", response_text)
    if numbers:
        return numbers[-1]

    return None


def format_demo_example(question, answer, rationale=None):
    """
    Format a demonstration example for the prompt.

    Args:
        question: Question text
        answer: Numeric answer
        rationale: Optional rationale/reasoning steps

    Returns:
        Formatted demo string
    """
    if rationale is None:
        # Just question and answer
        return f"Q: {question}\nA: {answer}"
    else:
        # Question, rationale, and answer
        return f"Q: {question}\nA: {rationale} The answer is {answer}."
