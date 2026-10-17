import re

def analyze_prompt(prompt: str) -> dict:
    text = prompt.lower()
    words = len(re.findall(r"\b\w+\b", text))
    code_signals = ["code", "debug", "algorithm", "implement", "python", "c++", "typescript", "sql", "api"]
    reasoning_signals = ["prove", "compare", "architect", "design", "why", "trade-off", "tradeoff", "analyze", "derive", "step by step"]
    complex_hits = sum(s in text for s in reasoning_signals) + min(words // 120, 3)
    code_hits = sum(s in text for s in code_signals)
    if complex_hits >= 3 or words > 500:
        complexity = "high"
    elif complex_hits >= 1 or code_hits >= 2 or words > 120:
        complexity = "medium"
    else:
        complexity = "low"
    return {"complexity": complexity, "words": words, "code_intent": code_hits >= 2, "reasoning_intent": complex_hits >= 1}
