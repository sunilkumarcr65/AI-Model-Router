from .models import MODELS, ModelSpec


def score_model(
    model: ModelSpec,
    complexity: str,
    quality_target: float,
    latency_budget_ms: int,
    privacy: str,
    prompt: str,
    stats: dict | None = None,
) -> tuple[float, list[str]]:

    stats = stats or {}

    score = 0.0
    reasons = []

    # ---------------------------------------------------------
    # 1. QUALITY
    # ---------------------------------------------------------

    if model.quality >= quality_target:
        score += 40
        reasons.append("meets quality target")
    else:
        quality_gap = quality_target - model.quality
        score -= quality_gap * 100
        reasons.append("below quality target")

    # ---------------------------------------------------------
    # 2. STATIC SPEED
    # ---------------------------------------------------------

    if model.speed >= 0.85:
        score += 15
        reasons.append("strong static speed profile")
    elif model.speed >= 0.70:
        score += 10
        reasons.append("reasonable static speed profile")
    else:
        score += 5

    # ---------------------------------------------------------
    # 3. REQUEST COMPLEXITY
    # ---------------------------------------------------------

    if complexity == "high":
        score += model.quality * 20
        reasons.append("high complexity favors stronger reasoning")

    elif complexity == "low":
        score += model.speed * 20
        reasons.append("low complexity favors faster model")

    else:
        score += (model.quality + model.speed) * 8
        reasons.append("balanced request favors quality and speed")

    # ---------------------------------------------------------
    # 4. REQUEST / MODEL STRENGTH MATCH
    # ---------------------------------------------------------

    prompt_lower = prompt.lower()

    if any(strength in prompt_lower for strength in model.strengths):
        score += 5
        reasons.append("model strengths match request")

    # ---------------------------------------------------------
    # 5. PRIVACY
    # ---------------------------------------------------------

    if privacy == "sensitive":
        # Current free providers are public cloud APIs.
        # We don't want to send sensitive requests to them.
        score -= 1000
        reasons.append("sensitive mode blocks public free endpoints")

    # ---------------------------------------------------------
    # 6. HISTORICAL TELEMETRY
    # ---------------------------------------------------------

    request_count = stats.get("request_count", 0)
    success_rate = stats.get("success_rate")
    avg_latency = stats.get("avg_latency_ms")
    p95_latency = stats.get("p95_latency_ms")

    if request_count > 0:

        # -----------------------------
        # Reliability
        # -----------------------------

        if success_rate is not None:

            if success_rate >= 0.98:
                score += 15
                reasons.append(
                    f"high observed reliability ({success_rate * 100:.1f}%)"
                )

            elif success_rate >= 0.90:
                score += 8
                reasons.append(
                    f"good observed reliability ({success_rate * 100:.1f}%)"
                )

            elif success_rate >= 0.75:
                score -= 5
                reasons.append(
                    f"moderate observed reliability ({success_rate * 100:.1f}%)"
                )

            else:
                score -= 20
                reasons.append(
                    f"poor observed reliability ({success_rate * 100:.1f}%)"
                )

        # -----------------------------
        # Observed latency
        # -----------------------------

        if avg_latency is not None:

            if avg_latency <= latency_budget_ms:
                score += 15
                reasons.append(
                    f"observed average latency fits budget ({avg_latency:.0f} ms)"
                )
            else:
                # Penalize models that have historically exceeded
                # the requested latency budget.
                excess_ratio = (
                    avg_latency - latency_budget_ms
                ) / max(latency_budget_ms, 1)

                penalty = min(excess_ratio * 30, 30)

                score -= penalty

                reasons.append(
                    f"observed average latency exceeds budget ({avg_latency:.0f} ms)"
                )

        # -----------------------------
        # P95 latency
        # -----------------------------

        if p95_latency is not None:

            if p95_latency <= latency_budget_ms:
                score += 10
                reasons.append(
                    f"observed p95 latency fits budget ({p95_latency:.0f} ms)"
                )
            else:
                score -= 10
                reasons.append(
                    f"observed p95 latency exceeds budget ({p95_latency:.0f} ms)"
                )

    else:
        # Cold-start handling.
        # We don't punish a model simply because we have
        # never observed it before.
        reasons.append("cold-start model; using static metadata")

    return score, reasons


def rank_models(
    prompt: str,
    complexity: str,
    quality_target: float,
    latency_budget_ms: int,
    privacy: str,
    configured_ids: set[str],
    model_stats: dict[str, dict] | None = None,
):

    model_stats = model_stats or {}

    ranked = []

    for model in MODELS:

        if model.id not in configured_ids:
            continue

        stats = model_stats.get(model.id, {})

        score, reasons = score_model(
            model=model,
            complexity=complexity,
            quality_target=quality_target,
            latency_budget_ms=latency_budget_ms,
            privacy=privacy,
            prompt=prompt,
            stats=stats,
        )

        ranked.append(
            {
                "model": model,
                "score": round(score, 3),
                "reasons": reasons,
                "stats": stats,
            }
        )

    return sorted(
        ranked,
        key=lambda x: x["score"],
        reverse=True,
    )