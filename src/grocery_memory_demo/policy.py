from __future__ import annotations

from datetime import date

from grocery_memory_demo.memory import expired, is_pinned
from grocery_memory_demo.models import MemoryItem, Verdict

MISTAKE_COSTS = {"hard_constraint": 100, "dietary_pref": 10, "brand_taste": 2}
ASK_COST = 3


def decide(item: MemoryItem, today: date) -> Verdict:
    if expired(item, today):
        return Verdict.IGNORE
    if is_pinned(item.status, item.category):
        return Verdict.USE
    loss = MISTAKE_COSTS[item.category]
    harm_of_using = (1 - item.reliability) * loss
    harm_of_ignoring = item.reliability * loss
    if min(harm_of_using, harm_of_ignoring) > ASK_COST:
        return Verdict.ASK
    return Verdict.USE if harm_of_using < harm_of_ignoring else Verdict.IGNORE


def verdicts(memory: list[MemoryItem], today: date) -> dict[str, Verdict]:
    return {item.topic: decide(item, today) for item in memory}
