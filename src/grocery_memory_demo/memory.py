from __future__ import annotations

from datetime import date

from grocery_memory_demo.models import CATEGORIES, HARD_CONSTRAINT, MemoryItem, Order, Scope, Status

RELIABILITY_HALF_LIFE_DAYS = 140
INFERRED_EXPIRY_DAYS = 90
PINNED_RELIABILITY = 1.0


def is_pinned(status: Status, category: str) -> bool:
    return status is Status.EXPLICIT and category == HARD_CONSTRAINT


def build_item(claim: dict, history: list[Order], today: date) -> MemoryItem:
    category = _validated_category(claim)
    status = Status(claim["status"])
    evidence_refs = _known_evidence_refs(claim, history)
    item = MemoryItem(
        claim=claim["claim"],
        category=category,
        status=status,
        evidence_refs=evidence_refs,
        last_evidence=max(evidence_dates(evidence_refs, history)),
        reliability=0.0,
        expiry_days=None if is_pinned(status, category) else INFERRED_EXPIRY_DAYS,
        scope=Scope(claim["scope"]),
        topic=claim["topic"],
    )
    item.reliability = reliability(item, history, today)
    return item


def update(
    memory: list[MemoryItem], claims: list[dict], history: list[Order], today: date
) -> list[MemoryItem]:
    superseded = {_identity(item): item for item in memory}
    items = [build_item(claim, history, today) for claim in claims]
    _reject_duplicate_identities(items)
    for item in items:
        previous = superseded.get(_identity(item))
        if previous is not None:
            item.version = _next_version(previous, item)
    return items


def reliability(item: MemoryItem, history: list[Order], today: date) -> float:
    if is_pinned(item.status, item.category):
        return PINNED_RELIABILITY
    dates = evidence_dates(item.evidence_refs, history)
    if not dates:
        return 0.0
    observable_orders = [order for order in history if order.date >= min(dates)]
    support = len(dates) / len(observable_orders)
    return round(support * _recency_factor(item.last_evidence, today), 3)


def expired(item: MemoryItem, today: date) -> bool:
    if item.expiry_days is None:
        return False
    return _age_days(item.last_evidence, today) > item.expiry_days


def evidence_dates(refs: list[str], history: list[Order]) -> list[date]:
    dates_by_order = {order.id: order.date for order in history}
    return [dates_by_order[ref] for ref in refs if ref in dates_by_order]


def _recency_factor(last_evidence: date, today: date) -> float:
    return 0.5 ** (_age_days(last_evidence, today) / RELIABILITY_HALF_LIFE_DAYS)


def _age_days(last_evidence: date, today: date) -> int:
    return (today - last_evidence).days


def _next_version(previous: MemoryItem, item: MemoryItem) -> int:
    if _content(previous) == _content(item):
        return previous.version
    return previous.version + 1


def _identity(item: MemoryItem) -> tuple[str, str]:
    return item.category, item.topic


def _content(item: MemoryItem) -> tuple:
    return item.claim, item.status, item.scope, tuple(item.evidence_refs)


def _reject_duplicate_identities(items: list[MemoryItem]) -> None:
    identities = [_identity(item) for item in items]
    duplicates = sorted(identity for identity in set(identities) if identities.count(identity) > 1)
    if duplicates:
        raise ValueError(f"duplicate claim identities: {duplicates!r}")


def _validated_category(claim: dict) -> str:
    category = claim["category"]
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r} in claim {claim['claim']!r}")
    return category


def _known_evidence_refs(claim: dict, history: list[Order]) -> list[str]:
    known = {order.id for order in history}
    refs = sorted({ref for ref in claim["evidence_refs"] if ref in known})
    if not refs:
        raise ValueError(f"no known evidence orders for claim {claim['claim']!r}")
    return refs
