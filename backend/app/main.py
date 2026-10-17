
import time
import uuid
from pydantic import BaseModel, Field
from statistics import mean

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from .analyzer import analyze_prompt
from .budget import estimate_output_budget
from .config import settings
from .db import ModelAttempt, RequestLog, SessionLocal, init_db
from .models import MODELS
from .providers import ProviderError, call_model
from .quality import evaluate_response
from .router import rank_models


class ChatRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The user prompt to send to the AI router.",
        examples=["Explain how a stack works and give an example."],
    )
    quality_target: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum target quality score, from 0 to 1.",
    )
    latency_budget_ms: int = Field(
        default=15000,
        gt=0,
        description="Target latency budget in milliseconds.",
    )
    privacy: str = Field(
        default="standard",
        description="Privacy requirement used during model selection.",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Optional manual output-token override. Omit for automatic budgeting.",
    )
    preferred_model: str | None = Field(
        default=None,
        description="Optional model ID to prioritize.",
    )


app = FastAPI(
    title="AI Model Router",
    version="1.1.0",
    description=(
        "Intelligent multi-objective AI model routing gateway "
        "with adaptive telemetry, intelligent output budgeting, "
        "fallback, and quality-aware routing."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()


def configured_model_ids() -> set[str]:
    configured = set()

    if settings.google_api_key:
        configured.add("gemini-3.8-flash")

    if settings.openrouter_api_key:
        for model in MODELS:
            if model.api == "openrouter":
                configured.add(model.id)

    return configured


async def get_model_stats() -> dict[str, dict]:
    """
    Build historical telemetry for each model.

    Both successful and failed attempts are included.
    Failed attempts can have latency because ProviderError
    records the time spent before failure.
    """
    async with SessionLocal() as session:
        result = await session.execute(
            select(ModelAttempt)
        )
        attempts = result.scalars().all()

    stats: dict[str, dict] = {}

    for model in MODELS:
        model_attempts = [
            attempt
            for attempt in attempts
            if attempt.model_id == model.id
        ]

        total = len(model_attempts)

        if total == 0:
            continue

        successful = [
            attempt
            for attempt in model_attempts
            if attempt.success
        ]

        failed = [
            attempt
            for attempt in model_attempts
            if not attempt.success
        ]

        latencies = [
            attempt.latency_ms
            for attempt in model_attempts
            if attempt.latency_ms is not None
        ]

        fallback_count = sum(
            1
            for attempt in model_attempts
            if attempt.fallback
        )

        avg_latency = mean(latencies) if latencies else None
        sorted_latencies = sorted(latencies)

        if sorted_latencies:
            index = min(
                len(sorted_latencies) - 1,
                max(0, int(len(sorted_latencies) * 0.95)),
            )
            p95_latency = sorted_latencies[index]
        else:
            p95_latency = None

        stats[model.id] = {
            "request_count": total,
            "attempt_count": total,
            "successful_attempts": len(successful),
            "failed_attempts": len(failed),
            "success_rate": len(successful) / total,
            "failure_rate": len(failed) / total,
            "avg_latency_ms": (
                round(avg_latency, 2)
                if avg_latency is not None
                else None
            ),
            "p95_latency_ms": (
                round(p95_latency, 2)
                if p95_latency is not None
                else None
            ),
            "fallback_count": fallback_count,
        }

    return stats


@app.get("/health")
async def health():
    configured = configured_model_ids()

    return {
        "status": "ok",
        "configured_models": len(configured),
        "total_models": len(MODELS),
    }


@app.get("/api/models")
async def list_models():
    configured = configured_model_ids()

    return {
        "models": [
            {
                "id": model.id,
                "name": model.name,
                "provider": model.provider,
                "api": model.api,
                "quality": model.quality,
                "speed": model.speed,
                "context": model.context,
                "input_cost": model.input_cost,
                "output_cost": model.output_cost,
                "configured": model.id in configured,
                "strengths": list(model.strengths),
            }
            for model in MODELS
        ]
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    request_data = request.model_dump()

    prompt = str(request_data.get("prompt", "")).strip()

    if not prompt:
        return {"error": "Prompt cannot be empty."}

    quality_target = float(
        request_data.get("quality_target", 0.85)
    )

    latency_budget_ms = int(
        request_data.get("latency_budget_ms", 15000)
    )

    privacy = str(
        request_data.get("privacy", "standard")
    )

    # =========================================================
    # OPTIONAL MANUAL TOKEN OVERRIDE
    # =========================================================

    manual_max_tokens = request_data.get("max_tokens")

    if manual_max_tokens is not None:
        manual_max_tokens = int(manual_max_tokens)

        if manual_max_tokens <= 0:
            return {
                "error": "max_tokens must be greater than 0."
            }

    # FIX: JSON null becomes Python None, so handle it directly.
    preferred_model = (
        request_data.get("preferred_model") or ""
    ).strip()

    request_key = str(uuid.uuid4())

    # =========================================================
    # ANALYZE REQUEST
    # =========================================================

    analysis = analyze_prompt(prompt)
    complexity = analysis["complexity"]

    # =========================================================
    # INTELLIGENT OUTPUT BUDGET
    # =========================================================

    budget = estimate_output_budget(
        prompt=prompt,
        complexity=complexity,
    )

    estimated_max_tokens = budget["max_tokens"]
    budget_reasons = budget["reasons"]

    if manual_max_tokens is not None:
        max_tokens = manual_max_tokens
        token_budget_mode = "manual_override"
        token_budget_reasons = [
            "manual max_tokens override supplied by client"
        ]
    else:
        max_tokens = estimated_max_tokens
        token_budget_mode = "automatic"
        token_budget_reasons = budget_reasons

    # =========================================================
    # LOAD HISTORICAL TELEMETRY
    # =========================================================

    model_stats = await get_model_stats()
    configured_ids = configured_model_ids()

    # =========================================================
    # RANK MODELS
    # =========================================================

    ranking = rank_models(
        prompt=prompt,
        complexity=complexity,
        quality_target=quality_target,
        latency_budget_ms=latency_budget_ms,
        privacy=privacy,
        configured_ids=configured_ids,
        model_stats=model_stats,
    )

    if not ranking:
        return {
            "error": "No configured AI models are available."
        }

    # =========================================================
    # PREFERRED MODEL
    # =========================================================

    if preferred_model:
        preferred = next(
            (
                item
                for item in ranking
                if item["model"].id == preferred_model
            ),
            None,
        )

        if preferred:
            ranking = [preferred] + [
                item
                for item in ranking
                if item["model"].id != preferred_model
            ]

    # =========================================================
    # ATTEMPT MODELS
    # =========================================================

    errors = []
    quality_failures = []

    successful_attempt = None
    answer = None
    usage = None
    selected_model = None
    selected_latency = None
    successful_quality = None
    attempt_number = 0

    async with SessionLocal() as session:
        for item in ranking:
            model = item["model"]

            attempt_number += 1
            is_fallback = attempt_number > 1
            started = time.perf_counter()

            # -------------------------------------------------
            # CALL PROVIDER
            # -------------------------------------------------

            try:
                (
                    candidate_answer,
                    candidate_usage,
                    latency,
                ) = await call_model(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )

            except ProviderError as exc:
                latency = exc.latency_ms

                if latency is None:
                    latency = (
                        time.perf_counter() - started
                    ) * 1000

                error_message = str(exc)

                errors.append({
                    "model": model.id,
                    "type": "provider_failure",
                    "error": error_message,
                })

                failed_attempt = ModelAttempt(
                    request_key=request_key,
                    model_id=model.id,
                    provider=model.provider,
                    latency_ms=latency,
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error=error_message,
                    fallback=is_fallback,
                )

                session.add(failed_attempt)
                await session.flush()
                continue

            # -------------------------------------------------
            # QUALITY EVALUATION + DEBUGGING
            # -------------------------------------------------

            print("\n" + "=" * 70)
            print(f"[QUALITY DEBUG] Model: {model.id}")
            print(f"[QUALITY DEBUG] Answer: {candidate_answer!r}")
            print(f"[QUALITY DEBUG] Usage: {candidate_usage}")
            print(f"[QUALITY DEBUG] Max tokens: {max_tokens}")
            print("=" * 70)

            candidate_quality = evaluate_response(
                prompt=prompt,
                answer=candidate_answer,
                output_tokens=candidate_usage.get(
                    "output_tokens",
                    0,
                ),
                max_tokens=max_tokens,
            )

            print(
                f"[QUALITY DEBUG] Result: {candidate_quality}"
            )
            print("=" * 70 + "\n")

            # -------------------------------------------------
            # QUALITY FAILURE
            # -------------------------------------------------

            if not candidate_quality["passed"]:
                quality_error = (
                    "Quality evaluation failed: "
                    + "; ".join(candidate_quality["reasons"])
                )

                quality_failures.append({
                    "model": model.id,
                    "quality_score": candidate_quality["score"],
                    "reasons": candidate_quality["reasons"],
                })

                errors.append({
                    "model": model.id,
                    "type": "quality_failure",
                    "error": quality_error,
                })

                failed_attempt = ModelAttempt(
                    request_key=request_key,
                    model_id=model.id,
                    provider=model.provider,
                    latency_ms=latency,
                    input_tokens=candidate_usage.get(
                        "input_tokens",
                        0,
                    ),
                    output_tokens=candidate_usage.get(
                        "output_tokens",
                        0,
                    ),
                    success=False,
                    error=quality_error,
                    fallback=is_fallback,
                )

                session.add(failed_attempt)
                await session.flush()

                # Try the next model.
                continue

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------

            successful_attempt = ModelAttempt(
                request_key=request_key,
                model_id=model.id,
                provider=model.provider,
                latency_ms=latency,
                input_tokens=candidate_usage.get(
                    "input_tokens",
                    0,
                ),
                output_tokens=candidate_usage.get(
                    "output_tokens",
                    0,
                ),
                success=True,
                error=None,
                fallback=is_fallback,
            )

            session.add(successful_attempt)
            await session.flush()

            answer = candidate_answer
            usage = candidate_usage
            selected_model = model
            selected_latency = latency
            successful_quality = candidate_quality

            break

        # =====================================================
        # NO MODEL PRODUCED A VALID RESPONSE
        # =====================================================

        if successful_attempt is None:
            await session.commit()

            return {
                "request_key": request_key,
                "answer": None,
                "error": (
                    "All configured models failed "
                    "provider or quality checks."
                ),
                "complexity": complexity,
                "token_budget": {
                    "mode": token_budget_mode,
                    "max_tokens": max_tokens,
                    "estimated_max_tokens": estimated_max_tokens,
                    "reasons": token_budget_reasons,
                },
                "attempt_number": attempt_number,
                "fallbacks_attempted": max(
                    0,
                    attempt_number - 1,
                ),
                "errors": errors,
                "quality_failures": quality_failures,
                "ranking": [
                    {
                        "model": item["model"].id,
                        "name": item["model"].name,
                        "provider": item["model"].provider,
                        "score": item["score"],
                        "reasons": item["reasons"],
                        "stats": item["stats"],
                    }
                    for item in ranking
                ],
            }

        # =====================================================
        # SAVE REQUEST-LEVEL LOG
        # =====================================================

        request_log = RequestLog(
            request_text=prompt,
            selected_model=selected_model.id,
            provider=selected_model.provider,
            complexity=complexity,
            quality_target=quality_target,
            latency_budget_ms=latency_budget_ms,
            privacy=privacy,
            latency_ms=selected_latency,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            success=True,
            error=None,
        )

        session.add(request_log)
        await session.commit()

    # =========================================================
    # RESPONSE
    # =========================================================

    return {
        "request_key": request_key,
        "answer": answer,
        "selected_model": {
            "id": selected_model.id,
            "name": selected_model.name,
            "provider": selected_model.provider,
        },
        "complexity": complexity,
        "token_budget": {
            "mode": token_budget_mode,
            "max_tokens": max_tokens,
            "estimated_max_tokens": estimated_max_tokens,
            "reasons": token_budget_reasons,
        },
        "latency_ms": round(selected_latency, 2),
        "usage": usage,
        "quality": {
            "passed": successful_quality["passed"],
            "score": successful_quality["score"],
            "reasons": successful_quality["reasons"],
        },
        "attempt_number": attempt_number,
        "fallbacks_attempted": max(0, attempt_number - 1),
        "errors": errors,
        "quality_failures": quality_failures,
        "ranking": [
            {
                "model": item["model"].id,
                "name": item["model"].name,
                "provider": item["model"].provider,
                "score": item["score"],
                "reasons": item["reasons"],
                "stats": item["stats"],
            }
            for item in ranking
        ],
    }


@app.get("/api/metrics")
async def metrics():
    async with SessionLocal() as session:
        result = await session.execute(
            select(RequestLog)
        )
        requests = result.scalars().all()

        attempt_result = await session.execute(
            select(ModelAttempt)
        )
        attempts = attempt_result.scalars().all()

    total_requests = len(requests)

    successful_requests = sum(
        1
        for request in requests
        if request.success
    )

    success_rate = (
        successful_requests / total_requests * 100
        if total_requests
        else 0
    )

    latencies = [
        request.latency_ms
        for request in requests
        if request.latency_ms is not None
    ]

    avg_latency = mean(latencies) if latencies else 0

    model_distribution = {}

    for request in requests:
        model_distribution[request.selected_model] = (
            model_distribution.get(request.selected_model, 0) + 1
        )

    successful_attempts = sum(
        1
        for attempt in attempts
        if attempt.success
    )

    failed_attempts = sum(
        1
        for attempt in attempts
        if not attempt.success
    )

    fallback_attempts = sum(
        1
        for attempt in attempts
        if attempt.fallback
    )

    model_stats = {}

    for model in MODELS:
        model_attempts = [
            attempt
            for attempt in attempts
            if attempt.model_id == model.id
        ]

        if not model_attempts:
            continue

        total = len(model_attempts)

        successful = sum(
            1
            for attempt in model_attempts
            if attempt.success
        )

        failed = total - successful

        model_latencies = sorted([
            attempt.latency_ms
            for attempt in model_attempts
            if attempt.latency_ms is not None
        ])

        if model_latencies:
            avg = mean(model_latencies)
            p95_index = min(
                len(model_latencies) - 1,
                int(len(model_latencies) * 0.95),
            )
            p95 = model_latencies[p95_index]
        else:
            avg = None
            p95 = None

        model_stats[model.id] = {
            "attempt_count": total,
            "successful_attempts": successful,
            "failed_attempts": failed,
            "success_rate": successful / total,
            "failure_rate": failed / total,
            "avg_latency_ms": (
                round(avg, 2) if avg is not None else None
            ),
            "p95_latency_ms": (
                round(p95, 2) if p95 is not None else None
            ),
            "fallback_count": sum(
                1
                for attempt in model_attempts
                if attempt.fallback
            ),
        }

    total_attempts = len(attempts)

    attempt_failure_rate = (
        failed_attempts / total_attempts * 100
        if total_attempts
        else 0
    )

    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "success_rate": round(success_rate, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "model_distribution": model_distribution,
        "attempts": {
            "total": total_attempts,
            "successful": successful_attempts,
            "failed": failed_attempts,
            "failure_rate": round(attempt_failure_rate, 2),
            "fallback_attempts": fallback_attempts,
        },
        "model_stats": model_stats,
        "estimated_cost_usd": 0.0,
    }
