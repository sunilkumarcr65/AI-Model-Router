from datetime import datetime

from sqlalchemy import String, Float, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)

from .config import settings


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True
    )

    request_text: Mapped[str] = mapped_column(Text)

    selected_model: Mapped[str] = mapped_column(String(160))

    provider: Mapped[str] = mapped_column(String(80))

    complexity: Mapped[str] = mapped_column(String(30))

    quality_target: Mapped[float] = mapped_column(Float)

    latency_budget_ms: Mapped[int] = mapped_column(Integer)

    privacy: Mapped[str] = mapped_column(String(30))

    latency_ms: Mapped[float] = mapped_column(Float)

    input_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    output_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    success: Mapped[bool] = mapped_column(
        Boolean,
        default=True
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )


class ModelAttempt(Base):
    """
    Records every individual model attempt.

    One user request can generate multiple attempts:

        Gemini       -> failed
        Lightning    -> success

    Both attempts are stored here.
    """

    __tablename__ = "model_attempts"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True
    )

    request_key: Mapped[str] = mapped_column(
        String(80),
        index=True
    )

    model_id: Mapped[str] = mapped_column(
        String(160),
        index=True
    )

    provider: Mapped[str] = mapped_column(
        String(80)
    )

    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    input_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    output_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    success: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    fallback: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )


engine = create_async_engine(
    settings.database_url,
    future=True
)

SessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession
)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all
        )