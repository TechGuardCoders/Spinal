"""Shared workload definitions for Spinal + Batcher-style traffic.

Same mixed-realistic philosophy: short chat, medium explanation, code
generation, long-form — weighted like real traffic.
"""
import random

PROMPT_CLASSES = {
    "short": [
        "What is 2+2? Answer in one word.",
        "Name three primary colors.",
        "Say hello in French, one word.",
    ],
    "medium": [
        "Explain continuous batching in one paragraph to a bank auditor.",
        "Summarize the trade-offs between MIG and time-slicing GPUs in two sentences.",
        "What does p95 latency mean and why do SREs prefer it to the mean?",
        "Explain prefix caching to a developer who has never run an inference server.",
    ],
    "code": [
        "Write a Python function that checks whether a string is a palindrome, with type hints and a docstring.",
        "Write a JSON object with keys model, tokens_in, tokens_out, cost_usd for an inference bill of $0.42 at 1M tokens.",
        "One-liner in bash: find all files over 100MB in /var and print their sizes.",
        "Write a SQL query: top 5 customers by total order value, with a CTE.",
    ],
    "long": [
        "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry: type hints, docstrings, and a short design rationale.",
        "Draft a professional email to a client explaining that a scheduled maintenance window will move to Saturday night, and why.",
        "Explain how KV cache size scales with context length and why it constrains concurrency, in three paragraphs.",
    ],
}

WEIGHTS = {"short": 0.25, "medium": 0.35, "code": 0.25, "long": 0.15}


def pick_prompt(rng: random.Random):
    r = rng.random()
    acc = 0.0
    for cls, w in WEIGHTS.items():
        acc += w
        if r <= acc:
            pool = PROMPT_CLASSES[cls]
            return cls, pool[rng.randrange(len(pool))]
    cls = "medium"
    pool = PROMPT_CLASSES[cls]
    return cls, pool[rng.randrange(len(pool))]
