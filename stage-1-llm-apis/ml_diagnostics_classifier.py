"""
ML Diagnostics Classifier
Stage 1 — AI Engineering Journey

Uses the Anthropic Claude API to classify ML model issues
and return structured diagnostic information.

Author: Oluwanifemi Afolayan
Medium: https://medium.com/@nifemiafolayanofficial
GitHub: https://github.com/oluwanifemitech
"""

import json
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment


SYSTEM_PROMPT = """You are an ML diagnostics assistant.
Always respond with valid JSON only. No markdown, no explanation, no backticks.
Schema: {
  "issue": string,
  "severity": "low" | "medium" | "high",
  "likely_cause": string,
  "recommended_fix": string,
  "confidence": float between 0 and 1
}"""


def classify_ml_issue(description: str) -> dict:
    """
    Classify an ML model issue and return a structured diagnosis.

    Args:
        description: Natural language description of the model problem

    Returns:
        Dictionary with issue, severity, likely_cause, recommended_fix, confidence
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": description}]
    )

    if response.stop_reason != "end_turn":
        raise ValueError(f"Unexpected stop reason: {response.stop_reason}")

    return json.loads(response.content[0].text)


def print_diagnosis(issue: str, result: dict) -> None:
    """Pretty print a diagnosis result."""
    severity_icons = {"low": "🟡", "medium": "🟠", "high": "🔴"}
    icon = severity_icons.get(result["severity"], "⚪")

    print(f"\n{'='*60}")
    print(f"Issue: {issue}")
    print(f"{'='*60}")
    print(f"  Diagnosis:   {result['issue']}")
    print(f"  Severity:    {icon} {result['severity'].upper()}")
    print(f"  Cause:       {result['likely_cause']}")
    print(f"  Fix:         {result['recommended_fix'][:120]}...")
    print(f"  Confidence:  {result['confidence']:.0%}")


if __name__ == "__main__":
    test_cases = [
        "My model gets 99% training accuracy but 60% validation accuracy",
        "Loss is not decreasing after epoch 5 despite low learning rate",
        "Model performs well on weekday data but poorly on weekends"
    ]

    print("ML Diagnostics Classifier — Powered by Claude API")
    print("Stage 1 | AI Engineering Journey\n")

    for issue in test_cases:
        result = classify_ml_issue(issue)
        print_diagnosis(issue, result)

    print(f"\n{'='*60}")
    print("Done. All diagnoses returned valid structured JSON.")
