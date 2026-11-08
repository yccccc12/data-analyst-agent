"""Shared test helpers: a scripted fake model, so agent tests never touch the network."""
import json

import pytest

from app import llm


class Item:
    """Mimics an OpenAI SDK output item: attribute access plus model_dump()."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def model_dump(self, exclude_none=True):
        return dict(self.__dict__)


class Response:
    def __init__(self, output, text=""):
        self.output = output
        self.output_text = text


def tool_call(name, arguments, call_id=None):
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return Item(type="function_call", name=name, arguments=args, call_id=call_id or f"call_{name}")


def final(text):
    return Response(
        [Item(type="message", role="assistant", content=[{"type": "output_text", "text": text}])],
        text,
    )


def calls(*tool_calls):
    return Response([Item(type="reasoning", summary=[]), *tool_calls])


@pytest.fixture
def fake_llm(monkeypatch):
    """Usage: fake_llm([step, step, ...]).  A step is a Response, or a function(history) -> Response.

    The returned object records every call in `.inputs` (a snapshot of the history sent).
    """
    def install(script):
        steps = list(script)
        seen = {"inputs": []}

        def fake_call_model(input_items, tools=None):
            seen["inputs"].append(list(input_items))
            step = steps.pop(0)
            return step(input_items) if callable(step) else step

        monkeypatch.setattr(llm, "call_model", fake_call_model)
        return type("Fake", (), {"inputs": seen["inputs"], "remaining": lambda self: len(steps)})()

    return install
