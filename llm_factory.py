import os
from importlib import import_module
from typing import Optional

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter


def _build_rate_limiter() -> InMemoryRateLimiter:
    requests_per_second = float(os.getenv("LLM_REQUESTS_PER_SECOND", "0.3"))
    check_every_n_seconds = float(os.getenv("LLM_RATE_LIMIT_CHECK_SECONDS", "0.1"))
    return InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=check_every_n_seconds,
    )


def create_llm(provider: Optional[str] = None, temperature: Optional[float] = None):
    """Create and return an LLM chat model based on environment configuration.

    Supported providers:
    - mistral_api: Mistral hosted API via langchain-mistralai
    - ollama_local: Local models served through Ollama via langchain-ollama
    - vllm: Local models served through vLLM via langchain-openai
    """
    load_dotenv()

    selected_provider = (provider or os.getenv("LLM_PROVIDER", "mistral_api")).strip().lower()
    selected_temperature = (
        float(os.getenv("LLM_TEMPERATURE", "0.1")) if temperature is None else temperature
    )

    if selected_provider == "mistral_api":
        from langchain_mistralai import ChatMistralAI

        model_name = os.getenv("MISTRAL_LLM_MODEL", "mistral-large-latest")
        use_rate_limiter = os.getenv("LLM_USE_RATE_LIMITER", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        kwargs = {
            "model_name": model_name,
            "temperature": selected_temperature,
        }
        if use_rate_limiter:
            kwargs["rate_limiter"] = _build_rate_limiter()

        return ChatMistralAI(**kwargs)

    if selected_provider in {"vllm", "vllm_local"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "LLM_PROVIDER=vllm requires 'langchain-openai'. Install it with: pip install langchain-openai"
            ) from exc

        model_name = os.getenv("VLLM_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

        kwargs = {
            "model": model_name,
            "openai_api_base": base_url,
            "openai_api_key": "none",
            "temperature": selected_temperature,
        }
        return ChatOpenAI(**kwargs)

    if selected_provider in {"ollama_local", "ollama"}:
        try:
            ChatOllama = import_module("langchain_ollama").ChatOllama
        except ImportError as exc:
            raise ImportError(
                "LLM_PROVIDER=ollama_local requires 'langchain-ollama'. Install it with: pip install langchain-ollama"
            ) from exc

        model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        kwargs = {
            "model": model_name,
            "base_url": base_url,
            "temperature": selected_temperature,
        }

        num_ctx = os.getenv("OLLAMA_NUM_CTX")
        if num_ctx:
            kwargs["num_ctx"] = int(num_ctx)

        return ChatOllama(**kwargs)

    raise ValueError(
        "Unsupported LLM_PROVIDER. Use one of: 'mistral_api', 'ollama_local', 'vllm'."
    )
