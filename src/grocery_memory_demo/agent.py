from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from datetime import date

from grocery_memory_demo.checks import hard_constraint_violations
from grocery_memory_demo.llm import MODEL, client
from grocery_memory_demo.models import (
    Arm,
    MemoryItem,
    Order,
    OrderItem,
    Product,
    ToolCall,
    Transcript,
)
from grocery_memory_demo.policy import decide
from grocery_memory_demo.prompts import (
    MEMORY_SECTION,
    SHOPPING_PROMPT,
    SIGNALS_SECTION,
    TOOLS,
    VERDICT_SECTION,
)

MAX_TURNS = 8
MAX_TOKENS = 16000
EFFORT = "high"


def run_arm(
    arm: Arm,
    request: str,
    customer_answer: str,
    memory: list[MemoryItem],
    catalogue: list[Product],
    history: list[Order],
    today: date,
    on_model_turn: Callable[[int], None] | None = None,
) -> Transcript:
    transcript = Transcript(arm=arm, request=request)
    system = system_prompt(arm, memory, history, today)
    messages: list[dict] = [{"role": "user", "content": request}]

    for turn in range(1, MAX_TURNS + 1):
        if on_model_turn is not None:
            on_model_turn(turn)
        response = client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        requested_tools = [block for block in response.content if block.type == "tool_use"]
        if not requested_tools:
            transcript.reply = _reply_text(response)
            break
        results = []
        for block in requested_tools:
            result = _run_tool(block.name, block.input, transcript, catalogue, customer_answer)
            transcript.tool_calls.append(ToolCall(block.name, dict(block.input), result))
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": results})
    else:
        transcript.reply = f"Stopped after reaching the {MAX_TURNS}-turn limit."

    transcript.violations = hard_constraint_violations(transcript.basket, memory, catalogue)
    return transcript


def system_prompt(arm: Arm, memory: list[MemoryItem], history: list[Order], today: date) -> str:
    sections = [
        SHOPPING_PROMPT,
        SIGNALS_SECTION.format(orders=len(history), signals=render_signals(history)),
    ]
    if arm is Arm.WITH_MEMORY:
        sections.append(MEMORY_SECTION.format(memory=render_memory(memory)))
    if arm is Arm.WITH_VERDICTS:
        sections.append(VERDICT_SECTION.format(memory=render_memory(memory, today)))
    return "\n\n".join(sections)


def render_signals(history: list[Order]) -> str:
    orders_containing = Counter(
        product_id for order in history for product_id in {item.product_id for item in order.items}
    )
    ranked = sorted(orders_containing.items(), key=lambda pair: (-pair[1], pair[0]))
    return "\n".join(f"- {product_id}: {count}" for product_id, count in ranked)


def render_memory(memory: list[MemoryItem], today: date | None = None) -> str:
    return "\n".join(_memory_line(item, today) for item in memory)


def _memory_line(item: MemoryItem, today: date | None) -> str:
    facts = f"{item.category}, {item.status.value}, reliability {item.reliability:.2f}"
    line = f"- {item.claim} [{facts}]"
    if today is None:
        return line
    return f"{line} -> {decide(item, today).value}"


def _run_tool(
    name: str,
    arguments: dict,
    transcript: Transcript,
    catalogue: list[Product],
    customer_answer: str,
) -> str:
    if name == "search_catalogue":
        return _search_catalogue(arguments["query"], catalogue)
    if name == "add_to_cart":
        return _add_to_cart(arguments["product_id"], int(arguments["qty"]), transcript, catalogue)
    if name == "ask_customer":
        transcript.questions.append(arguments["question"])
        return customer_answer
    raise ValueError(f"agent called unknown tool {name!r}")


def _search_catalogue(query: str, catalogue: list[Product]) -> str:
    words = [word for word in _words(query) if len(word) >= 3]
    matches = [product for product in catalogue if _matches(product, words)]
    if not matches:
        return f"No products match {query!r}."
    return "\n".join(_product_line(product) for product in matches)


def _matches(product: Product, words: list[str]) -> bool:
    product_words = set(_words(" ".join([product.id, product.name, *product.tags])))
    return any(word in product_words for word in words)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _product_line(product: Product) -> str:
    allergens = ", ".join(product.allergens) or "none"
    stock = "in stock" if product.in_stock else "OUT OF STOCK"
    return (
        f"{product.id} | {product.name} | tags: {', '.join(product.tags)} "
        f"| allergens: {allergens} | EUR {product.price:.2f} | {stock}"
    )


def _add_to_cart(
    product_id: str, qty: int, transcript: Transcript, catalogue: list[Product]
) -> str:
    product = next((entry for entry in catalogue if entry.id == product_id), None)
    if product is None:
        return f"No product {product_id!r} in the catalogue. Use an id from a search result."
    if not product.in_stock:
        return f"{product.name} is out of stock. Nothing was added."
    transcript.basket.append(OrderItem(product_id, qty))
    return f"Added {qty} x {product.name}."


def _reply_text(response) -> str:
    return "\n".join(block.text for block in response.content if block.type == "text").strip()
