from app.quality import evaluate_response

tests = [
    {
        "name": "Truncated stack explanation",
        "prompt": "Explain what a stack is and give an example.",
        "answer": (
            "A stack is a fundamental concept in computer science. "
            "It is a linear"
        ),
        "tokens": 16,
        "max_tokens": 512,
        "expected": False,
    },
    {
        "name": "Incomplete linear data answer",
        "prompt": "Explain what a stack is and give an example.",
        "answer": (
            "### What is a Stack?\n\n"
            "In computer science, a stack is a linear data"
        ),
        "tokens": 19,
        "max_tokens": 512,
        "expected": False,
    },
    {
        "name": "Incomplete stack example ending in Push",
        "prompt": "Explain what a stack is and give an example.",
        "answer": (
            "Example:*\n"
            "*   Imagine an empty stack.\n"
            "*   Push"
        ),
        "tokens": 16,
        "max_tokens": 512,
        "expected": False,
    },
    {
        "name": "Unclosed code fence",
        "prompt": "Explain Python code.",
        "answer": (
            "Here is an example:\n"
            "```python\n"
            "print('hello')"
        ),
        "tokens": 10,
        "max_tokens": 512,
        "expected": False,
    },
    {
        "name": "Valid arithmetic answer",
        "prompt": "What is 2 + 2?",
        "answer": "4",
        "tokens": 1,
        "max_tokens": 256,
        "expected": True,
    },
    {
        "name": "Implementation without code",
        "prompt": "Implement binary search in Python.",
        "answer": (
            "Binary search repeatedly divides a sorted "
            "search interval in half."
        ),
        "tokens": 12,
        "max_tokens": 1024,
        "expected": False,
    },
    {
        "name": "Implementation with code",
        "prompt": "Implement binary search in Python.",
        "answer": (
            "Binary search finds a target in a sorted list.\n"
            "```python\n"
            "def binary_search(arr, target):\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        if arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1\n"
            "```\n"
            "This returns the target index, or -1 if absent."
        ),
        "tokens": 80,
        "max_tokens": 1024,
        "expected": True,
    },
    {
        "name": "Valid example with complete list",
        "prompt": "Explain what a stack is and give an example.",
        "answer": (
            "A stack is a data structure that follows "
            "last in, first out (LIFO). For example, "
            "push 10 onto an empty stack, then push 20. "
            "The top element is now 20."
        ),
        "tokens": 35,
        "max_tokens": 512,
        "expected": True,
    },
]

failures = 0

for test in tests:
    result = evaluate_response(
        prompt=test["prompt"],
        answer=test["answer"],
        output_tokens=test["tokens"],
        max_tokens=test["max_tokens"],
    )

    passed = result["passed"] == test["expected"]

    print(f'{"PASS" if passed else "FAIL"}: {test["name"]}')
    print(f'  Expected: {test["expected"]}')
    print(f'  Actual:   {result["passed"]}')
    print(f'  Score:    {result["score"]}')
    print(f'  Reasons:  {result["reasons"]}')

    if not passed:
        failures += 1

print(f"\n{len(tests) - failures}/{len(tests)} tests passed")

if failures:
    raise SystemExit(1)
