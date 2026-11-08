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


def require_llm_config():
    """Fail early with a clear message instead of silently calling the wrong endpoint."""
    missing = [
        name for name, value in [("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY), ("LLM_BASE_URL", BASE_URL), ("LLM_MODEL", MODEL)]
        if not value
    ]

    if missing:
        raise RuntimeError(f"Missing required settings in .env: {', '.join(missing)}")
