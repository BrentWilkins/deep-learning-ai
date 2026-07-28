import os

from dotenv import find_dotenv, load_dotenv


def load_env() -> None:
    """Load environment variables from the nearest .env file."""
    _ = load_dotenv(find_dotenv())


def get_model_api_key() -> str | None:
    """Return the optional OpenRouter API key used by alternate examples."""
    load_env()
    return os.getenv("OPENROUTER_API_KEY")
