from __future__ import annotations

import re

from grocery_memory_demo.memory import is_pinned
from grocery_memory_demo.models import MemoryItem, OrderItem, Product, Violation


def hard_constraint_violations(
    basket: list[OrderItem], memory: list[MemoryItem], catalogue: list[Product]
) -> list[Violation]:
    forbidden = forbidden_allergens(memory, catalogue)
    allergens_by_product = {product.id: product.allergens for product in catalogue}
    return [
        Violation(entry.product_id, allergen, claim)
        for entry in basket
        for allergen, claim in forbidden.items()
        if allergen in _allergens_of(entry.product_id, allergens_by_product)
    ]


def forbidden_allergens(memory: list[MemoryItem], catalogue: list[Product]) -> dict[str, str]:
    vocabulary = sorted({allergen for product in catalogue for allergen in product.allergens})
    return {
        allergen: item.claim
        for item in memory
        if is_pinned(item.status, item.category)
        for allergen in vocabulary
        if _mentions(item.claim, allergen)
    }


def _mentions(claim: str, allergen: str) -> bool:
    words = re.sub(r"[^a-z0-9]+", " ", claim.lower())
    phrase = re.escape(allergen.replace("_", " "))
    return re.search(rf"\b{phrase}s?\b", words) is not None


def _allergens_of(product_id: str, allergens_by_product: dict[str, list[str]]) -> list[str]:
    if product_id not in allergens_by_product:
        raise ValueError(f"basket contains unknown product {product_id!r}")
    return allergens_by_product[product_id]
