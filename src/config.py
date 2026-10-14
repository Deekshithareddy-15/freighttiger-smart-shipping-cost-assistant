"""
Configuration module for FreightTiger Smart Shipping Cost Assistant.
Loads environment variables and sets system-wide constants.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""
    
    # Project paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    
    # Dataset files
    SHIPMENT_RECORDS_PATH: Path = DATA_DIR / "shipment_records.csv"
    CONTEXT_NOTES_PATH: Path = DATA_DIR / "context_notes.csv"
    SAMPLE_OUTPUT_FORMAT_PATH: Path = DATA_DIR / "sample_output_format.csv"
    
    # Default output files
    OUTPUT_CSV_PATH: Path = OUTPUT_DIR / "output.csv"
    OUTPUT_ALL_WEEKS_CSV_PATH: Path = OUTPUT_DIR / "output_all_weeks.csv"
    TOKEN_COST_LOG_PATH: Path = OUTPUT_DIR / "token_cost_log.json"
    REPRODUCIBILITY_REPORT_PATH: Path = OUTPUT_DIR / "reproducibility_report.txt"
    
    # Anomaly Detection Thresholds
    # A route-week is flagged as candidate anomaly if cost is rising and exceeds either:
    # 1. Trailing 8-week own history rolling average by >= OWN_HISTORY_THRESHOLD_PCT
    # 2. Peer same-week average (same route_type) by >= SIMILAR_ROUTES_THRESHOLD_PCT
    OWN_HISTORY_THRESHOLD_PCT: float = Field(default=20.0, description="Percentage threshold above own 8w rolling avg")
    SIMILAR_ROUTES_THRESHOLD_PCT: float = Field(default=20.0, description="Percentage threshold above peer avg")
    
    # Baseline Rolling Window
    TRAILING_WEEKS_WINDOW: int = 8
    
    # AI / LLM Configuration
    # Options: 'local_deterministic' (default, free, 100% reproducible, zero API key required)
    # or 'openai' / 'groq' / 'ollama' when API keys are configured.
    LLM_PROVIDER: str = Field(default="local_deterministic", description="LLM provider: local_deterministic, openai, groq, ollama")
    LLM_MODEL: str = Field(default="local-guardrail-v1", description="Model identifier")
    LLM_API_KEY: str = Field(default="", description="API key for cloud LLMs (optional)")
    LLM_BASE_URL: str = Field(default="", description="Custom base URL for OpenAI-compatible or Ollama endpoints")
    LLM_TEMPERATURE: float = Field(default=0.0, description="Temperature strictly 0.0 for reproducibility")
    
    # Vector DB
    CHROMA_PERSIST_DIR: Path = BASE_DIR / ".chroma_db"
    CHROMA_COLLECTION_NAME: str = "freight_context_notes"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
