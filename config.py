"""Configuration management for the YouTube RAG Pipeline."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent


@dataclass
class Config:
    """Central configuration for the pipeline."""

    # Paths
    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma_db")
    state_file: str = str(BASE_DIR / "data" / "pipeline_state.json")
    log_file: str = str(BASE_DIR / "logs" / "pipeline.log")
    sessions_dir: str = str(BASE_DIR / "data" / "sessions")
    personas_dir: str = str(BASE_DIR / "data" / "personas")

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Text splitting
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Scheduler
    update_interval_minutes: int = 60

    # MCP
    mcp_host: str = "localhost"
    mcp_port: int = 8765

    # Logging
    log_level: str = "INFO"

    # Google Gemini (transcript extraction)
    gemini_api_key: str = ""

    def __post_init__(self):
        """Override defaults with environment variables if present."""
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", self.chroma_persist_dir)
        self.state_file = os.getenv("STATE_FILE", self.state_file)
        self.log_file = os.getenv("LOG_FILE", self.log_file)
        self.sessions_dir = os.getenv("SESSIONS_DIR", self.sessions_dir)
        self.personas_dir = os.getenv("PERSONAS_DIR", self.personas_dir)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", self.chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", self.chunk_overlap))
        self.update_interval_minutes = int(
            os.getenv("UPDATE_INTERVAL_MINUTES", self.update_interval_minutes)
        )
        self.mcp_host = os.getenv("MCP_SERVER_HOST", self.mcp_host)
        self.mcp_port = int(os.getenv("MCP_SERVER_PORT", self.mcp_port))
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", self.gemini_api_key)

        # Ensure directories exist
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.sessions_dir).mkdir(parents=True, exist_ok=True)
        Path(self.personas_dir).mkdir(parents=True, exist_ok=True)


config = Config()
