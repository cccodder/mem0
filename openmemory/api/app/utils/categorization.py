import logging
import os
from typing import List, Optional

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


def _get_openai_client() -> OpenAI:
    """Initialize OpenAI client using configuration from database or environment."""
    from app.database import SessionLocal
    from app.models import Config as ConfigModel

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url: Optional[str] = None
    model = "gpt-4o-mini"

    try:
        db = SessionLocal()
        try:
            db_config = db.query(ConfigModel).filter(ConfigModel.key == "main").first()
            if db_config and "mem0" in db_config.value:
                mem0_config = db_config.value["mem0"]
                if "llm" in mem0_config and mem0_config["llm"]:
                    llm_config = mem0_config["llm"]
                    if "config" in llm_config:
                        config = llm_config["config"]
                        # Get model name
                        if "model" in config:
                            model = config["model"]
                        # Get API key
                        if "api_key" in config:
                            api_key_value = config["api_key"]
                            if api_key_value and api_key_value.startswith("env:"):
                                env_var = api_key_value.split(":", 1)[1]
                                api_key = os.environ.get(env_var, api_key)
                            else:
                                api_key = api_key_value
                        # Get base URL
                        if "openai_base_url" in config:
                            base_url = config["openai_base_url"]
        finally:
            db.close()
    except Exception as e:
        logging.warning(f"Failed to load config from database: {e}, using defaults")

    client_kwargs = {"api_key": api_key, "timeout": 60.0, "max_retries": 3}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs), model


_openai_client: Optional[OpenAI] = None
_model: str = "gpt-4o-mini"


def _get_client_and_model():
    """Get or initialize the OpenAI client and model."""
    global _openai_client, _model
    if _openai_client is None:
        _openai_client, _model = _get_openai_client()
    return _openai_client, _model


def reset_categorization_client():
    """Reset the categorization client to force reinitialization."""
    global _openai_client, _model
    _openai_client = None
    _model = "gpt-4o-mini"


class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    client, model = _get_client_and_model()
    completion = None
    try:
        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory}
        ]

        # Let OpenAI handle the pydantic parsing directly
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=MemoryCategories,
            temperature=0
        )

        parsed: MemoryCategories = completion.choices[0].message.parsed
        return [cat.strip().lower() for cat in parsed.categories]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        try:
            if completion and completion.choices:
                logging.debug(f"[DEBUG] Raw response: {completion.choices[0].message.content}")
        except Exception as debug_e:
            logging.debug(f"[DEBUG] Could not extract raw response: {debug_e}")
        raise
