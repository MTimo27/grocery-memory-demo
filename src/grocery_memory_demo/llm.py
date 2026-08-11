from __future__ import annotations

from functools import cache

from anthropic import Anthropic
from dotenv import load_dotenv

MODEL = "claude-sonnet-5"


@cache
def client() -> Anthropic:
    load_dotenv()
    return Anthropic()
