import re


def estimate_output_budget(prompt: str, complexity: str) -> dict:
    """
    Estimate how many output tokens a model should be allowed to generate.

    This is a heuristic first version.
    Later, we can replace/augment it with a learned estimator using telemetry.
    """

    text = prompt.lower().strip()
    words = len(re.findall(r"\b\w+\b", text))

    budget = 128
    reasons = []

    # ---------------------------------------------------------
    # 1. Complexity
    # ---------------------------------------------------------

    if complexity == "low":
        budget = 256
        reasons.append("low-complexity request")

    elif complexity == "medium":
        budget = 768
        reasons.append("medium-complexity request")

    elif complexity == "high":
        budget = 1536
        reasons.append("high-complexity request")

    # ---------------------------------------------------------
    # 2. Coding requests
    # ---------------------------------------------------------

    coding_signals = [
        "write code",
        "implement",
        "implementation",
        "code",
        "program",
        "function",
        "class",
        "python",
        "c++",
        "javascript",
        "typescript",
        "sql",
        "api",
        "debug",
    ]

    code_hits = sum(signal in text for signal in coding_signals)

    if code_hits >= 1:
        budget += 256
        reasons.append("coding-related request")

    # ---------------------------------------------------------
    # 3. Detailed explanation / reasoning
    # ---------------------------------------------------------

    reasoning_signals = [
        "explain in detail",
        "step by step",
        "analyze",
        "analysis",
        "compare",
        "comparison",
        "trade-off",
        "tradeoff",
        "architecture",
        "design",
        "why",
        "derive",
        "prove",
    ]

    reasoning_hits = sum(signal in text for signal in reasoning_signals)

    if reasoning_hits >= 1:
        budget += 384
        reasons.append("reasoning or detailed explanation requested")

    # ---------------------------------------------------------
    # 4. Multiple requested tasks
    # ---------------------------------------------------------

    task_signals = [
        " and ",
        "also",
        "additionally",
        "then",
        "along with",
        "include",
        "including",
    ]

    task_hits = sum(signal in text for signal in task_signals)

    if task_hits >= 2:
        budget += 256
        reasons.append("multiple tasks detected")

    # ---------------------------------------------------------
    # 5. Explicit long-answer requests
    # ---------------------------------------------------------

    long_answer_signals = [
        "comprehensive",
        "thorough",
        "in depth",
        "in-depth",
        "detailed",
        "complete explanation",
        "full explanation",
        "deep explanation",
        "everything",
    ]

    if any(signal in text for signal in long_answer_signals):
        budget += 512
        reasons.append("user requested detailed output")

    # ---------------------------------------------------------
    # 6. Prompt length
    # ---------------------------------------------------------

    if words > 300:
        budget += 384
        reasons.append("long input prompt")

    elif words > 150:
        budget += 192
        reasons.append("moderately long input prompt")

    # ---------------------------------------------------------
    # 7. Simple factual questions
    # ---------------------------------------------------------

    simple_patterns = [
        r"^what is .+\??$",
        r"^who is .+\??$",
        r"^when was .+\??$",
        r"^where is .+\??$",
        r"^is .+\??$",
        r"^are .+\??$",
        r"^can .+\??$",
        r"^what are .+\??$",
    ]

    simple_question = any(
        re.match(pattern, text)
        for pattern in simple_patterns
    )

    if simple_question and complexity == "low" and code_hits == 0:
        budget = min(budget, 256)
        reasons.append("simple factual question")

    # ---------------------------------------------------------
    # 8. Explanation requests
    # ---------------------------------------------------------

    explanation_signals = [
        "explain",
        "describe",
        "how does",
        "how do",
        "why does",
        "why do",
    ]

    if any(signal in text for signal in explanation_signals):
        budget = max(budget, 512)
        reasons.append("explanation requires sufficient output budget")

    # ---------------------------------------------------------
    # 9. Minimum and maximum safety bounds
    # ---------------------------------------------------------

    budget = max(128, budget)
    budget = min(4096, budget)

    return {
        "max_tokens": budget,
        "estimated_output_tokens": budget,
        "reasons": reasons,
    }