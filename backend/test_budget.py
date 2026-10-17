from app.budget import estimate_output_budget


tests = [
    (
        "What is 2 + 2?",
        "low",
    ),
    (
        "Explain what a stack is and give an example.",
        "low",
    ),
    (
        "Explain binary search and implement it in Python.",
        "medium",
    ),
    (
        "Design a distributed cache and explain the architecture, "
        "sharding, replication, consistency, and failure handling.",
        "high",
    ),
    (
        "Analyze this system step by step, compare the alternatives, "
        "and explain the trade-offs in detail.",
        "high",
    ),
]


for prompt, complexity in tests:
    result = estimate_output_budget(prompt, complexity)

    print("\nPROMPT:")
    print(prompt)

    print("\nBUDGET:")
    print(result["max_tokens"])

    print("REASONS:")
    for reason in result["reasons"]:
        print(" -", reason)

    print("-" * 60)