from __future__ import annotations

from grocery_memory_demo.models import CATEGORIES, Scope, Status

EXTRACTION_PROMPT = """You read a household's grocery order history and propose candidate
memory claims about that household. You do NOT score them: reliability, expiry and
recency are computed downstream in code. Never output a confidence or a score.

Read both the line items AND the free-text notes on orders. Notes are where customers
state things explicitly ("never send X", "I'm allergic to Y").

Propose one claim per distinct pattern. Include patterns that look one-off, seasonal or
outdated - downstream scoring decides what survives. Do not merge unrelated patterns.

Every claim states something the household does, prefers or requires. Never claim that
something stopped, ended, lapsed or is absent, and never cite orders as evidence for a
product not being in them. A pattern that has ended is still stated in the present tense
and cites only the orders it did appear in; the scoring code represents its ending as
decayed reliability, and a claim asserting the ending would override that.

Field guidance:
  "claim":         one readable sentence, present tense, e.g. "Prefers lactose-free dairy"
  "category":      hard_constraint = an explicitly stated safety/allergy/never-send rule;
                   never infer one from purchases. dietary_pref = diet or nutrition pattern;
                   brand_taste = product or brand liking
  "status":        "explicit" if the customer stated it in a note, else "inferred"
  "topic":         short snake_case key for the pattern, e.g. "lactose_free_dairy".
                   It is the identity of the claim across runs - keep it stable.
  "scope":         "member" is about one person, "occasion" is a one-off event
  "evidence_refs": order ids where you observed it, e.g. ["order_03", "order_05"].
                   For an explicit statement, the single order carrying the note."""

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "status": {"type": "string", "enum": [status.value for status in Status]},
                    "topic": {"type": "string"},
                    "scope": {"type": "string", "enum": [scope.value for scope in Scope]},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "category", "status", "topic", "scope", "evidence_refs"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

SHOPPING_PROMPT = """You are a grocery-shopping assistant. A customer asks you to put products in
their basket for this week's delivery.

How to work:
- Search the catalogue before adding anything. Product ids must come from search results.
- Add each product with add_to_cart.
- Stay inside what the request asks for. Do not pad the basket with extras.
- When you have finished adding, reply in two or three sentences saying what you added and why."""

SIGNALS_SECTION = """Purchase signals for this household, aggregated over the order history -
how many of the {orders} orders contained each product:

{signals}"""

MEMORY_SECTION = """What we remember about this household. Each line carries a reliability score
between 0 and 1, computed from how often and how recently the pattern appeared:

{memory}"""

VERDICT_SECTION = """What we remember about this household. Each line carries a reliability score
and a verdict computed from it:

{memory}

The verdicts are decisions, not suggestions:
- use: treat the claim as true and shop accordingly.
- ignore: do not act on the claim. It is too weak or too stale to trust.
- ask: do not act on the claim until you have put it to the customer with ask_customer and read
  the answer. Ask one short question and then shop according to what they say."""

TOOLS = [
    {
        "name": "search_catalogue",
        "description": (
            "Search the product catalogue by keyword. Returns matching products with their id, "
            "tags, allergens, price and stock status. Search before every add_to_cart call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords, e.g. 'lactose free milk' or 'bread'.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_to_cart",
        "description": (
            "Add one product to the basket. The product id must come from a search result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "qty": {"type": "integer"},
            },
            "required": ["product_id", "qty"],
        },
    },
    {
        "name": "ask_customer",
        "description": (
            "Ask the customer one short question and read their answer. Use this when a memory "
            "claim carries the verdict 'ask'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
]
