from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    provider: str
    api: str
    quality: float
    speed: float
    context: int
    input_cost: float
    output_cost: float
    strengths: tuple[str, ...]
    requires_key: str


MODELS = [
    ModelSpec(
        "gemini-3.8-flash",
        "Gemini 3.8 Flash",
        "Google",
        "gemini",
        0.95,
        0.95,
        1048576,
        0.0,
        0.0,
        ("reasoning", "coding", "general", "fast"),
        "GOOGLE_API_KEY",
    ),

    ModelSpec(
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "NVIDIA Nemotron 3 Ultra",
        "OpenRouter / NVIDIA",
        "openrouter",
        0.97,
        0.76,
        1000000,
        0.0,
        0.0,
        ("reasoning", "agents", "coding", "long-context"),
        "OPENROUTER_API_KEY",
    ),

    ModelSpec(
        "nvidia/nemotron-3.5-lightning:free",
        "NVIDIA Nemotron 3.5 Lightning",
        "OpenRouter / NVIDIA",
        "openrouter",
        0.91,
        0.90,
        262144,
        0.0,
        0.0,
        ("fast", "reasoning", "general", "coding"),
        "OPENROUTER_API_KEY",
    ),

    ModelSpec(
        "cohere/north-mini-code:free",
        "Cohere North Mini Code",
        "OpenRouter / Cohere",
        "openrouter",
        0.91,
        0.82,
        262144,
        0.0,
        0.0,
        ("coding", "programming", "debugging", "code"),
        "OPENROUTER_API_KEY",
    ),

    ModelSpec(
        "nvidia/nemotron-3-super-120b-a12b:free",
        "NVIDIA Nemotron 3 Super 120B",
        "OpenRouter / NVIDIA",
        "openrouter",
        0.94,
        0.78,
        262144,
        0.0,
        0.0,
        ("reasoning", "coding", "analysis", "general"),
        "OPENROUTER_API_KEY",
    ),
]


BY_ID = {m.id: m for m in MODELS}