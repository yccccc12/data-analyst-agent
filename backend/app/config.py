import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent.parent

# LLMs service provider config
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")
ENABLE_THINKING = os.getenv("LLM_ENABLE_THINKING", "true").lower() == "true"
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "60"))


DB_PATH = (BACKEND_DIR / os.getenv("DB_PATH", "sample.db")).resolve()
CHARTS_DIR = BACKEND_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_ITERATIONS = 10  # hard stop for the agent loop
MAX_ROWS = 1000      # max rows any query may return
PREVIEW_ROWS = 20    # rows of a result sent back to the model
QUERY_TIMEOUT_S = 5

# Housekeeping so a long-running server does not grow without limit
SESSION_TTL_S = float(os.getenv("SESSION_TTL_S", "3600"))      # idle sessions are dropped after this
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "100"))           # least recently used go first
CHART_TTL_S = float(os.getenv("CHART_TTL_S", "86400"))         # chart PNGs older than this are deleted
CLEANUP_INTERVAL_S = 300                                       # how often chart cleanup may run


def require_llm_config():
    """Fail early with a clear message instead of silently calling the wrong endpoint."""
    missing = [
        name for name, value in [("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY), ("LLM_BASE_URL", BASE_URL), ("LLM_MODEL", MODEL)]
        if not value
    ]

    if missing:
        raise RuntimeError(f"Missing required settings in .env: {', '.join(missing)}")
