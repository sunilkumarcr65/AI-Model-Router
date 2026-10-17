import time

import httpx

from .config import settings
from .models import ModelSpec


class ProviderError(Exception):
    """
    Error raised when an AI provider fails.

    latency_ms stores the actual time spent before the
    provider failure occurred. This allows the adaptive
    router to learn from failed attempts as well as
    successful attempts.
    """

    def __init__(
        self,
        message: str,
        latency_ms: float | None = None,
    ):
        super().__init__(message)
        self.latency_ms = latency_ms


async def call_model(
    model: ModelSpec,
    prompt: str,
    max_tokens: int,
) -> tuple[str, dict, float]:

    started = time.perf_counter()

    try:
        # =====================================================
        # GEMINI
        # =====================================================

        if model.api == "gemini":

            if not settings.google_api_key:
                latency_ms = (
                    time.perf_counter() - started
                ) * 1000

                raise ProviderError(
                    "GOOGLE_API_KEY is not configured",
                    latency_ms=latency_ms,
                )

            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1beta/models/{model.id}:generateContent"
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                },
            }

            headers = {
                "x-goog-api-key": (
                    settings.google_api_key
                ),
            }

        # =====================================================
        # OPENROUTER
        # =====================================================

        elif model.api == "openrouter":

            if not settings.openrouter_api_key:
                latency_ms = (
                    time.perf_counter() - started
                ) * 1000

                raise ProviderError(
                    "OPENROUTER_API_KEY is not configured",
                    latency_ms=latency_ms,
                )

            url = (
                "https://openrouter.ai/api/v1/"
                "chat/completions"
            )

            payload = {
                "model": model.id,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }

            headers = {
                "Authorization": (
                    f"Bearer {settings.openrouter_api_key}"
                ),
                "Content-Type": "application/json",
                "X-Title": "AI Model Router",
            }

        # =====================================================
        # UNKNOWN PROVIDER
        # =====================================================

        else:

            latency_ms = (
                time.perf_counter() - started
            ) * 1000

            raise ProviderError(
                f"Unsupported provider API: {model.api}",
                latency_ms=latency_ms,
            )

        # =====================================================
        # HTTP REQUEST
        # =====================================================

        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds
        ) as client:

            response = await client.post(
                url,
                json=payload,
                headers=headers,
            )

            # -------------------------------------------------
            # HTTP ERROR
            # -------------------------------------------------

            if response.status_code >= 400:

                latency_ms = (
                    time.perf_counter() - started
                ) * 1000

                raise ProviderError(
                    (
                        f"{model.provider} returned "
                        f"HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    ),
                    latency_ms=latency_ms,
                )

            # -------------------------------------------------
            # Parse JSON
            # -------------------------------------------------

            try:
                data = response.json()

            except ValueError as exc:

                latency_ms = (
                    time.perf_counter() - started
                ) * 1000

                raise ProviderError(
                    (
                        f"{model.provider} returned "
                        "invalid JSON"
                    ),
                    latency_ms=latency_ms,
                ) from exc

    # =========================================================
    # TIMEOUT
    # =========================================================

    except httpx.TimeoutException as exc:

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        raise ProviderError(
            f"{model.provider} timed out",
            latency_ms=latency_ms,
        ) from exc

    # =========================================================
    # HTTP / CONNECTION ERROR
    # =========================================================

    except httpx.HTTPError as exc:

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        raise ProviderError(
            f"{model.provider} HTTP error: {exc}",
            latency_ms=latency_ms,
        ) from exc

    # =========================================================
    # PROVIDER ERROR
    # =========================================================

    except ProviderError:
        raise

    # =========================================================
    # UNEXPECTED ERROR
    # =========================================================

    except Exception as exc:

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        raise ProviderError(
            (
                f"{model.provider} unexpected "
                f"error: {exc}"
            ),
            latency_ms=latency_ms,
        ) from exc

    # =========================================================
    # SUCCESSFUL REQUEST LATENCY
    # =========================================================

    latency_ms = (
        time.perf_counter() - started
    ) * 1000

    # =========================================================
    # GEMINI RESPONSE
    # =========================================================

    if model.api == "gemini":

        try:
            candidates = data["candidates"]

            if not candidates:
                raise ValueError(
                    "Gemini returned no candidates"
                )

            content = candidates[0]["content"]

            parts = content.get(
                "parts",
                [],
            )

            answer = "".join(
                part.get("text", "")
                for part in parts
            )

            if not answer:
                raise ValueError(
                    "Gemini returned an empty response"
                )

            usage_metadata = data.get(
                "usageMetadata",
                {},
            )

            usage = {
                "input_tokens": usage_metadata.get(
                    "promptTokenCount",
                    0,
                ),
                "output_tokens": usage_metadata.get(
                    "candidatesTokenCount",
                    0,
                ),
            }

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:

            raise ProviderError(
                (
                    "Unexpected Gemini response: "
                    f"{exc}"
                ),
                latency_ms=latency_ms,
            ) from exc

    # =========================================================
    # OPENROUTER RESPONSE
    # =========================================================

    elif model.api == "openrouter":

        try:
            choices = data["choices"]

            if not choices:
                raise ValueError(
                    "OpenRouter returned no choices"
                )

            message = choices[0]["message"]

            answer = (
                message.get(
                    "content",
                    "",
                )
                or ""
            )

            if not answer:
                raise ValueError(
                    "OpenRouter returned an empty response"
                )

            usage_metadata = (
                data.get("usage")
                or {}
            )

            usage = {
                "input_tokens": usage_metadata.get(
                    "prompt_tokens",
                    0,
                ),
                "output_tokens": usage_metadata.get(
                    "completion_tokens",
                    0,
                ),
            }

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:

            raise ProviderError(
                (
                    "Unexpected OpenRouter response: "
                    f"{exc}"
                ),
                latency_ms=latency_ms,
            ) from exc

    # =========================================================
    # RETURN SUCCESS
    # =========================================================

    else:

        raise ProviderError(
            f"Unsupported provider API: {model.api}",
            latency_ms=latency_ms,
        )

    return answer, usage, latency_ms

