from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AILOGGER_", env_file=".env")

    database_url: str = "sqlite:///./ai_logger.db"
    master_passphrase: str | None = None


settings = Settings()
