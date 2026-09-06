from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Reads .env via pydantic-settings
class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    secret_key: str = "changeme"

    # --- Spotify OAuth ---
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    spotify_redirect_uri: str = "http://127.0.0.1:8000/auth/callback"

    # Where to send the browser after a successful OAuth callback
    frontend_url: str = "http://127.0.0.1:5173"

    # --- Postgres / Redis ---
    database_url: str = ""
    redis_url: str = ""

    # --- Gemini (genre tagging + cluster labeling) ---
    gemini_api_key: str = ""

@lru_cache
def get_settings() -> Settings:
    return Settings()
