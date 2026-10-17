from pydantic import BaseModel, Field
from typing import Literal

class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=50000)
    quality_target: float = Field(default=0.80, ge=0.50, le=1.0)
    latency_budget_ms: int = Field(default=15000, ge=500, le=120000)
    privacy: Literal["standard", "sensitive"] = "standard"
    max_tokens: int = Field(default=1200, ge=64, le=8192)
    preferred_model: str | None = None

class RouteResponse(BaseModel):
    answer: str
    selected_model: str
    provider: str
    complexity: str
    latency_ms: float
    usage: dict
    decision: dict

class ModelView(BaseModel):
    id: str
    name: str
    provider: str
    configured: bool
    quality: float
    speed: float
    context: int
    strengths: list[str]
    free: bool
