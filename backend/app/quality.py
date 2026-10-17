
import re
from typing import Any


def _word_count(text: str) -> int:
    """Count words in a response."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def _is_simple_math(prompt: str) -> bool:
    """Recognize simple arithmetic questions."""
    text = prompt.strip().lower()
    return bool(
        re.fullmatch(
            r"(?:what is\s+)?\d+\s*[+\-*/]\s*\d+\??",
            text,
        )
    )


def _is_code_request(prompt: str) -> bool:
    """Check whether the user explicitly asks for code."""
    text = prompt.lower()
    patterns = (
        r"\bwrite\b.*\bcode\b",
        r"\bimplement\b",
        r"\bcode\s+(?:for|to)\b",
        r"\bprogram\s+(?:for|to)\b",
        r"\bfunction\s+(?:for|that|to)\b",
        r"\bsolution\s+in\s+(?:python|java|c\+\+|javascript)\b",
        r"\bshow\s+(?:me\s+)?the\s+code\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _asks_for_example(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(example|examples|demonstrate|illustrate)\b",
            prompt,
            re.IGNORECASE,
        )
    )


def _asks_for_comparison(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(compare|comparison|difference between|"
            r"differences between|versus|vs\.?)\b",
            prompt,
            re.IGNORECASE,
        )
    )


def _has_unbalanced_delimiters(text: str) -> bool:
    """
    Detect common unclosed brackets while ignoring brackets inside
    ordinary quoted strings. This is a lightweight heuristic, not a
    complete parser for every programming language.
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    quote = None
    escaped = False

    for char in text:
        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if quote:
            if char == quote:
                quote = None
            continue

        if char in ("'", '"', "`"):
            quote = char
            continue

        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != pairs[char]:
                return True
            stack.pop()

    return bool(stack)


def _looks_truncated(
    answer: str,
    output_tokens: int = 0,
    max_tokens: int | None = None,
) -> bool:
    """
    Flag strong signs of truncation. Avoid treating every response that
    ends without a period as incomplete, because code, lists, headings,
    and short answers may legitimately do so.
    """
    text = answer.strip()

    if not text:
        return True

    # An unclosed Markdown code block is a strong signal.
    if text.count("```") % 2 != 0:
        return True

    # Unclosed brackets often indicate a cut-off code or expression.
    if _has_unbalanced_delimiters(text):
        return True

    # Clear dangling Markdown list/bullet markers.
    if re.search(r"(?:^|\n)\s*(?:[-*+]|\d+[.)])\s*$", text):
        return True

    # Common visibly unfinished endings, such as "Push", "for example:",
    # or "the following:".
    ending = re.sub(r"\s+", " ", text).rstrip()
    if re.search(
        r"(?:\b(?:such as|for example|including|"
        r"the following|as follows)\s*[:,-]?)$",
        ending,
        re.IGNORECASE,
    ):
        return True

    # A short list item ending with an unfinished action verb, e.g.:
    # "* Push" or "- Add".
    last_line = text.splitlines()[-1].strip()
    if re.fullmatch(
        r"(?:[-*+]\s*)?(?:push|pop|add|remove|insert|delete|"
        r"return|define|create|implement|calculate|"
        r"multiply|divide|compare|consider|use|take|"
        r"first|then)\s*[*`.,:;]*",
        last_line,
        re.IGNORECASE,
    ):
        return True

    # Detect a few common unfinished noun phrases without using a broad
    # rule that rejects ordinary answers ending in words like "the".
    if re.search(
        r"\b(?:a linear|an example of|the concept of|"
        r"the difference between|a type of|consists of|"
        r"such as a|such as an)\s*$",
        ending,
        re.IGNORECASE,
    ):
        return True

    # Token-limit exhaustion is suspicious only when the output also
    # lacks a normal ending. It is not enough by itself to reject an
    # answer, since provider token counts can be approximate.
    if max_tokens and max_tokens > 0 and output_tokens > 0:
        near_limit = output_tokens >= max_tokens * 0.98
        has_sentence_ending = bool(
            re.search(r'[.!?`}\])"\']\s*$', ending)
        )
        if near_limit and not has_sentence_ending:
            return True

    return False


def evaluate_response(
    prompt: str,
    answer: str,
    output_tokens: int = 0,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate an LLM response using conservative, explainable heuristics.

    Returns:
        {
            "passed": bool,
            "score": float,
            "reasons": list[str]
        }

    This is a heuristic evaluator, not a semantic truth verifier.
    """
    prompt = (prompt or "").strip()
    answer = (answer or "").strip()
    reasons: list[str] = []
    score = 1.0

    if not answer:
        return {
            "passed": False,
            "score": 0.0,
            "reasons": ["empty response"],
        }

    if _looks_truncated(answer, output_tokens, max_tokens):
        return {
            "passed": False,
            "score": 0.0,
            "reasons": ["response appears incomplete or truncated"],
        }

    # Simple arithmetic questions should receive a concise numeric answer.
    if _is_simple_math(prompt):
        expression = re.sub(
            r"^what is\s+",
            "",
            prompt.rstrip("?").strip(),
            flags=re.IGNORECASE,
        )
        expression = expression.replace(" ", "")
        expected = None

        try:
            match = re.fullmatch(r"(\d+)([+\-*/])(\d+)", expression)
            if match:
                left, operator, right = match.groups()
                a, b = int(left), int(right)

                if operator == "+":
                    expected = a + b
                elif operator == "-":
                    expected = a - b
                elif operator == "*":
                    expected = a * b
                elif operator == "/" and b != 0:
                    expected = a / b
        except (ValueError, ZeroDivisionError):
            expected = None

        if expected is not None:
            numeric_match = re.search(
                r"[-+]?\d+(?:\.\d+)?",
                answer.replace(",", ""),
            )
            if not numeric_match:
                return {
                    "passed": False,
                    "score": 0.0,
                    "reasons": ["expected a numeric answer"],
                }

            try:
                actual = float(numeric_match.group())
                if abs(actual - float(expected)) > 1e-9:
                    return {
                        "passed": False,
                        "score": 0.0,
                        "reasons": ["arithmetic answer does not match"],
                    }
            except ValueError:
                return {
                    "passed": False,
                    "score": 0.0,
                    "reasons": ["could not parse numeric answer"],
                }

            return {
                "passed": True,
                "score": 1.0,
                "reasons": [],
            }

    # Very short answers can be valid, so only flag them when the prompt
    # clearly requires a substantial response.
    if _word_count(answer) < 3 and not _is_simple_math(prompt):
        if _is_code_request(prompt):
            return {
                "passed": False,
                "score": 0.0,
                "reasons": ["requested implementation appears to be missing"],
            }

    if _is_code_request(prompt):
        has_code = bool(
            re.search(r"```", answer)
            or re.search(
                r"\b(def|class|function|const|let|var|"
                r"public static|#include|import)\b",
                answer,
            )
        )
        if not has_code:
            return {
                "passed": False,
                "score": 0.2,
                "reasons": ["requested code or implementation may be missing"],
            }

    if _asks_for_example(prompt):
        has_example = bool(
            re.search(
                r"\b(for example|e\.g\.|example:|consider|imagine|"
                r"suppose|such as|like this)\b",
                answer,
                re.IGNORECASE,
            )
        )
        if not has_example:
            score -= 0.15
            reasons.append("requested example may be missing")

    if _asks_for_comparison(prompt):
        has_comparison = bool(
            re.search(
                r"\b(whereas|while|unlike|compared with|"
                r"on the other hand|difference|similarit)\b",
                answer,
                re.IGNORECASE,
            )
        )
        if not has_comparison:
            score -= 0.15
            reasons.append("requested comparison may be incomplete")

    score = max(0.0, min(1.0, score))

    # Missing optional content lowers the score but does not automatically
    # trigger fallback. Strong structural truncation and missing code do.
    return {
        "passed": score >= 0.65,
        "score": round(score, 2),
        "reasons": reasons,
    }
