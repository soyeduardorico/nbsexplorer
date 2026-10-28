"""Project-wide configuration defaults."""
from pathlib import Path

from dotenv import load_dotenv

LOCAL_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_local_env(env_file: Path = LOCAL_ENV_FILE) -> None:
    load_dotenv(env_file, override=False)


load_local_env()

EE_PROJECT = "exeter-gee-research"
GEMINI_MODEL = "gemini-3.1-flash-lite"
