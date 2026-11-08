from functools import lru_cache

from openai import OpenAI

from . import config


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Plain OpenAI SDK pointed at DashScope (Qwen). No agent framework.

    Built lazily so importing the app (e.g. for /health) doesn't need credentials.
    """
    config.require_llm_config()
    return OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.BASE_URL,
        timeout=config.LLM_TIMEOUT_S,
    )


def call_model(input_items: list, tools: list | None = None):
    """One model call via the Responses API. Returns the raw response."""
    kwargs = {"model": config.MODEL, "input": input_items}
    if config.ENABLE_THINKING:
        kwargs["extra_body"] = {"enable_thinking": True}
    if tools:
        kwargs["tools"] = tools
    return get_client().responses.create(**kwargs)
