import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv()


def create_model() -> ChatDeepSeek:
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_base = os.getenv("DEEPSEEK_BASE_URL")

    if not model_name:
        raise RuntimeError("MODEL_NAME is not configured")

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    return ChatDeepSeek(
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=0,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )
