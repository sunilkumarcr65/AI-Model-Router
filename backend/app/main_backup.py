import uuid
from statistics import quantiles

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from .analyzer import analyze_prompt
from .config import settings
from .db import ModelAttempt, RequestLog, SessionLocal, init_db
from .models import BY_ID, MODELS
from .providers import ProviderError, call_model
from .router import rank_models
from .schemas import RouteRequest


app = FastAPI(
    title="AI Model Router",
    description=(
        "Intelligent AI gateway that routes requests across "
        "multiple real AI models using complexity, quality, "
        "latency, privacy, reliability, and historical telemetry."
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def startup():
    await init_db()


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():
    configured_models = 0

    for model in MODELS:
        if model.api == "gemini" and settings.google_api_key:
            configured_models += 1

        elif (
            model.api == "openrouter"
            and settings.openrouter_api_key
        ):
            configured_models += 1

    return {
        "status": "ok",
        "configured_models": configured_models,
        "total_models": len(MODELS),
    }


# =========================================================
# AVAILABLE MODELS
# =========================================================

@app.get("/api/models")
async def get_models():
    result = []

    for model in MODELS:
        if model.api == "gemini":
            configured = bool(settings.google_api_key)

        elif model.api == "openrouter":
            configured = bool(settings.openrouter_api_key)

        else:
            configured = False

        result.append(
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
                "strengths": list(model.strengths),
                "requires_key": model.requires_key,
                "configured": configured,
                "free": (
                    model.input_cost == 0
                    and model.output_cost == 0
                ),
            }
        )

    return result


# =========================================================
# MODEL TELEMETRY
# =========================================================

async def get_model_stats(session):
    result = await session.execute(
        select(ModelAttempt.model_id).distinct()
    )

    model_ids = [
        row[0]
        for row in result.all()
    ]

    stats = {}

    for model_id in model_ids:
        result = await session.execute(
            select(ModelAttempt)
            .where(
                ModelAttempt.model_id == model_id
            )
            .order_by(
                ModelAttempt.created_at.asc()
            )
        )

        attempts = result.scalars().all()

        if not attempts:
            continue

        total = len(attempts)

        successful = sum(
            1
            for attempt in attempts
            if attempt.success
        )

        failed = total - successful

        success_rate = (
            successful / total
            if total
            else 0
        )

        failure_rate = (
            failed / total
            if total
            else 0
        )

        # -------------------------------------------------
        # Include latency from successful AND failed
        # attempts.
        # -------------------------------------------------

        latencies = [
            attempt.latency_ms
            for attempt in attempts
            if attempt.latency_ms is not None
        ]

        avg_latency = (
            sum(latencies) / len(latencies)
            if latencies
            else None
        )

        p95_latency = None

        if latencies:
            if len(latencies) == 1:
                p95_latency = latencies[0]

            else:
                try:
                    p95_latency = quantiles(
                        latencies,
                        n=100,
                        method="inclusive",
                    )[94]

                except Exception:
                    p95_latency = max(latencies)

        fallback_count = sum(
            1
            for attempt in attempts
            if attempt.fallback
        )

        stats[model_id] = {
            "attempt_count": total,
            "successful_attempts": successful,
            "failed_attempts": failed,
            "success_rate": round(
                success_rate,
                4,
            ),
            "failure_rate": round(
                failure_rate,
                4,
            ),
            "avg_latency_ms": (
                round(
                    avg_latency,
                    2,
                )
                if avg_latency is not None
                else None
            ),
            "p95_latency_ms": (
                round(
                    p95_latency,
                    2,
                )
                if p95_latency is not None
                else None
            ),
            "fallback_count": fallback_count,
        }

    return stats


# =========================================================
# ROUTE REQUEST
# =========================================================

@app.post("/api/route")
async def route_request(request: RouteRequest):

    # -----------------------------------------------------
    # 1. Analyze request
    # -----------------------------------------------------

    analysis = analyze_prompt(
        request.prompt
    )

    complexity = analysis["complexity"]

    # -----------------------------------------------------
    # 2. Determine configured models
    # -----------------------------------------------------

    configured_ids = set()

    for model in MODELS:
        if (
            model.api == "gemini"
            and settings.google_api_key
        ):
            configured_ids.add(model.id)

        elif (
            model.api == "openrouter"
            and settings.openrouter_api_key
        ):
            configured_ids.add(model.id)

    if not configured_ids:
        raise HTTPException(
            status_code=503,
            detail="No AI model providers are configured.",
        )

    # -----------------------------------------------------
    # 3. Open database session
    # -----------------------------------------------------

    async with SessionLocal() as session:

        # -------------------------------------------------
        # 4. Get historical model statistics
        # -------------------------------------------------

        model_stats = await get_model_stats(
            session
        )

        # -------------------------------------------------
        # 5. Rank models
        # -------------------------------------------------

        ranked = rank_models(
            prompt=request.prompt,
            complexity=complexity,
            quality_target=request.quality_target,
            latency_budget_ms=request.latency_budget_ms,
            privacy=request.privacy,
            configured_ids=configured_ids,
            model_stats=model_stats,
        )

        # -------------------------------------------------
        # 6. Move preferred model to first position
        # -------------------------------------------------

        if request.preferred_model:
            preferred = BY_ID.get(
                request.preferred_model
            )

            if preferred is not None:
                preferred_entry = None

                for entry in ranked:
                    if entry["model"].id == preferred.id:
                        preferred_entry = entry
                        break

                if preferred_entry is not None:
                    ranked.remove(
                        preferred_entry
                    )

                    ranked.insert(
                        0,
                        preferred_entry
                    )

        if not ranked:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No eligible models are available "
                    "for the requested constraints."
                ),
            )

        # -------------------------------------------------
        # 7. Generate request ID
        # -------------------------------------------------

        request_key = str(
            uuid.uuid4()
        )

        errors = []

        final_model = None
        final_answer = None
        final_usage = None
        final_latency = None
        final_attempt_number = None

        # -------------------------------------------------
        # 8. Try models in ranked order
        # -------------------------------------------------

        for attempt_number, entry in enumerate(
            ranked,
            start=1,
        ):
            model = entry["model"]

            is_fallback = (
                attempt_number > 1
            )

            try:
                # -----------------------------------------
                # Call model
                # -----------------------------------------

                answer, usage, latency = await call_model(
                    model=model,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                )

                # -----------------------------------------
                # Record successful attempt
                # -----------------------------------------

                attempt = ModelAttempt(
                    request_key=request_key,
                    model_id=model.id,
                    provider=model.provider,
                    latency_ms=latency,
                    input_tokens=usage.get(
                        "input_tokens",
                        0,
                    ),
                    output_tokens=usage.get(
                        "output_tokens",
                        0,
                    ),
                    success=True,
                    error=None,
                    fallback=is_fallback,
                )

                session.add(attempt)

                # -----------------------------------------
                # Record final request
                # -----------------------------------------

                request_log = RequestLog(
                    request_text=request.prompt,
                    selected_model=model.id,
                    provider=model.provider,
                    complexity=complexity,
                    quality_target=request.quality_target,
                    latency_budget_ms=request.latency_budget_ms,
                    privacy=request.privacy,
                    latency_ms=latency,
                    input_tokens=usage.get(
                        "input_tokens",
                        0,
                    ),
                    output_tokens=usage.get(
                        "output_tokens",
                        0,
                    ),
                    success=True,
                    error=None,
                )

                session.add(request_log)

                await session.commit()

                final_model = model
                final_answer = answer
                final_usage = usage
                final_latency = latency
                final_attempt_number = (
                    attempt_number
                )

                break

            except ProviderError as exc:

                # -----------------------------------------
                # Record failed attempt
                # -----------------------------------------

                failed_attempt = ModelAttempt(
                    request_key=request_key,
                    model_id=model.id,
                    provider=model.provider,
                    latency_ms=exc.latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error=str(exc),
                    fallback=is_fallback,
                )

                session.add(
                    failed_attempt
                )

                await session.commit()

                errors.append(
                    {
                        "model": model.id,
                        "error": str(exc),
                        "latency_ms": (
                            round(
                                exc.latency_ms,
                                2,
                            )
                            if exc.latency_ms is not None
                            else None
                        ),
                    }
                )

                # -----------------------------------------
                # Try next model
                # -----------------------------------------

                continue

        # -------------------------------------------------
        # 9. All models failed
        # -------------------------------------------------

        if final_model is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": (
                        "All configured models failed."
                    ),
                    "request_key": request_key,
                    "errors": errors,
                },
            )

        # -------------------------------------------------
        # 10. Build ranking response
        # -------------------------------------------------

        ranking_response = []

        for entry in ranked:
            model = entry["model"]

            ranking_response.append(
                {
                    "model": model.id,
                    "name": model.name,
                    "provider": model.provider,
                    "score": entry["score"],
                    "reasons": entry["reasons"],
                    "stats": entry["stats"],
                }
            )

        # -------------------------------------------------
        # 11. Return response
        # -------------------------------------------------

        return {
            "request_key": request_key,
            "answer": final_answer,
            "selected_model": {
                "id": final_model.id,
                "name": final_model.name,
                "provider": final_model.provider,
            },
            "complexity": complexity,
            "latency_ms": round(
                final_latency,
                2,
            ),
            "usage": {
                "input_tokens": final_usage.get(
                    "input_tokens",
                    0,
                ),
                "output_tokens": final_usage.get(
                    "output_tokens",
                    0,
                ),
            },
            "attempt_number": final_attempt_number,
            "fallbacks_attempted": (
                final_attempt_number - 1
            ),
            "errors": errors,
            "ranking": ranking_response,
        }


# =========================================================
# METRICS
# =========================================================

@app.get("/api/metrics")
async def metrics():

    async with SessionLocal() as session:

        # -------------------------------------------------
        # Request metrics
        # -------------------------------------------------

        result = await session.execute(
            select(
                func.count(RequestLog.id)
            )
        )

        total_requests = (
            result.scalar() or 0
        )

        result = await session.execute(
            select(
                func.count(RequestLog.id)
            ).where(
                RequestLog.success.is_(True)
            )
        )

        successful_requests = (
            result.scalar() or 0
        )

        success_rate = (
            (
                successful_requests
                / total_requests
            )
            * 100
            if total_requests
            else 0
        )

        result = await session.execute(
            select(
                func.avg(
                    RequestLog.latency_ms
                )
            )
        )

        avg_latency = result.scalar()

        # -------------------------------------------------
        # Model distribution
        # -------------------------------------------------

        result = await session.execute(
            select(
                RequestLog.selected_model,
                func.count(RequestLog.id),
            )
            .group_by(
                RequestLog.selected_model
            )
        )

        model_distribution = {
            model_id: count
            for model_id, count in result.all()
        }

        # -------------------------------------------------
        # Attempt metrics
        # -------------------------------------------------

        result = await session.execute(
            select(
                func.count(ModelAttempt.id)
            )
        )

        attempt_total = (
            result.scalar() or 0
        )

        result = await session.execute(
            select(
                func.count(ModelAttempt.id)
            ).where(
                ModelAttempt.success.is_(True)
            )
        )

        attempt_success = (
            result.scalar() or 0
        )

        attempt_failures = (
            attempt_total
            - attempt_success
        )

        attempt_failure_rate = (
            (
                attempt_failures
                / attempt_total
            )
            * 100
            if attempt_total
            else 0
        )

        result = await session.execute(
            select(
                func.count(ModelAttempt.id)
            ).where(
                ModelAttempt.fallback.is_(True)
            )
        )

        fallback_attempts = (
            result.scalar() or 0
        )

        # -------------------------------------------------
        # Per-model statistics
        # -------------------------------------------------

        model_stats = await get_model_stats(
            session
        )

        # -------------------------------------------------
        # Estimated cost
        # -------------------------------------------------

        estimated_cost = 0.0

        result = await session.execute(
            select(RequestLog)
        )

        request_logs = (
            result.scalars().all()
        )

        for log in request_logs:
            model = BY_ID.get(
                log.selected_model
            )

            # Old model IDs may still exist in the
            # database. Do not delete them.
            if model is None:
                continue

            estimated_cost += (
                (
                    log.input_tokens
                    / 1_000_000
                )
                * model.input_cost
            )

            estimated_cost += (
                (
                    log.output_tokens
                    / 1_000_000
                )
                * model.output_cost
            )

        # -------------------------------------------------
        # Return metrics
        # -------------------------------------------------

        return {
            "total_requests": total_requests,
            "successful_requests": (
                successful_requests
            ),
            "success_rate": round(
                success_rate,
                2,
            ),
            "avg_latency_ms": (
                round(
                    avg_latency,
                    2,
                )
                if avg_latency is not None
                else 0
            ),
            "model_distribution": (
                model_distribution
            ),
            "attempts": {
                "total": attempt_total,
                "successful": attempt_success,
                "failed": attempt_failures,
                "failure_rate": round(
                    attempt_failure_rate,
                    2,
                ),
                "fallback_attempts": (
                    fallback_attempts
                ),
            },
            "model_stats": model_stats,
            "estimated_cost_usd": round(
                estimated_cost,
                6,
            ),
        }

