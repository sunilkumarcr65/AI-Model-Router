from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    google_api_key: str = ""
    openrouter_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///./router.db"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    request_timeout_seconds: float = 90.0
    redis_url: str = ""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @property
    def cors_list(self):
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

settings = Settings()
