from __future__ import annotations

import json

from grocery_memory_demo.llm import MODEL, client
from grocery_memory_demo.models import Order
from grocery_memory_demo.prompts import CLAIMS_SCHEMA, EXTRACTION_PROMPT

MAX_TOKENS = 16000
EFFORT = "high"


def extract_claims(history: list[Order], model: str = MODEL) -> list[dict]:
    response = client().messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=EXTRACTION_PROMPT,
        output_config={
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": CLAIMS_SCHEMA},
        },
        messages=[{"role": "user", "content": render_history(history)}],
    )
    return _parse_claims(response)


def render_history(history: list[Order]) -> str:
    return "\n".join(_render_order(order) for order in history)


def _render_order(order: Order) -> str:
    items = ", ".join(f"{item.product_id} x{item.qty}" for item in order.items)
    line = f"{order.id} (week {order.week}, {order.date.isoformat()}): {items}"
    if order.note:
        line += f'\n  note: "{order.note}"'
    return line


def _parse_claims(response) -> list[dict]:
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)["claims"]
