# Stage 1 — LLM APIs

Building with the Anthropic Claude API from scratch: conversation management, structured outputs, system prompt design, and parameter control.

## What's in here

### `ml_diagnostics_classifier.py`
An ML diagnostics tool that takes a description of a model problem and returns a structured JSON diagnosis — severity, likely cause, recommended fix, and confidence score.

**Example input:**
```
My model gets 99% training accuracy but 60% validation accuracy
```

**Example output:**
```json
{
  "issue": "Severe overfitting",
  "severity": "high",
  "likely_cause": "Model has learned noise and idiosyncrasies in training data",
  "recommended_fix": "Apply regularization techniques such as dropout or L1/L2 weight decay. Reduce model complexity. Implement early stopping.",
  "confidence": 0.95
}
```

## Key learnings

- The Claude API is **stateless** — conversation history must be managed manually on every call
- System prompts reduced token usage by **44%** compared to unguided prompts
- Always check `stop_reason` — a truncated response looks identical to a complete one
- `temperature=0` produces character-for-character identical output on every run

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"
python ml_diagnostics_classifier.py
```

## Companion article

[The Claude API from a Data Scientist's perspective — an honest first look](https://medium.com/@nifemiafolayanofficial)
