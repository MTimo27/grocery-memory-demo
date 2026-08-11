from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum

HARD_CONSTRAINT = "hard_constraint"
CATEGORIES = (HARD_CONSTRAINT, "dietary_pref", "brand_taste")


class Status(Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class Scope(Enum):
    HOUSEHOLD = "household"
    MEMBER = "member"
    OCCASION = "occasion"


class Verdict(Enum):
    USE = "use"
    IGNORE = "ignore"
    ASK = "ask"


class Arm(Enum):
    HISTORY_ONLY = "A"
    WITH_MEMORY = "B"
    WITH_VERDICTS = "C"


@dataclass
class Product:
    id: str
    name: str
    tags: list[str]
    allergens: list[str]
    price: float
    in_stock: bool


@dataclass
class OrderItem:
    product_id: str
    qty: int


@dataclass
class Order:
    id: str
    week: int
    date: date
    items: list[OrderItem]
    note: str | None = None


@dataclass
class MemoryItem:
    claim: str
    category: str
    status: Status
    evidence_refs: list[str]
    last_evidence: date
    reliability: float
    expiry_days: int | None
    scope: Scope
    topic: str = ""
    version: int = 1


@dataclass
class Violation:
    product_id: str
    allergen: str
    claim: str


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: str


@dataclass
class Transcript:
    arm: Arm
    request: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    basket: list[OrderItem] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    reply: str = ""
    violations: list[Violation] = field(default_factory=list)


@dataclass
class Metrics:
    correct: int = 0
    stale_errors: int = 0
    missed: int = 0
    clarifications: int = 0
    constraint_violations: int = 0


def to_dict(obj: object) -> dict:
    return {key: _encode(value) for key, value in asdict(obj).items()}


def _encode(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    return value


def product_from_dict(data: dict) -> Product:
    return Product(**data)


def order_from_dict(data: dict) -> Order:
    return Order(
        id=data["id"],
        week=data["week"],
        date=date.fromisoformat(data["date"]),
        items=[OrderItem(**item) for item in data["items"]],
        note=data.get("note"),
    )


def memory_item_from_dict(data: dict) -> MemoryItem:
    category = data["category"]
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r} in memory item {data['claim']!r}")
    return MemoryItem(
        claim=data["claim"],
        category=category,
        status=Status(data["status"]),
        evidence_refs=list(data["evidence_refs"]),
        last_evidence=date.fromisoformat(data["last_evidence"]),
        reliability=float(data["reliability"]),
        expiry_days=data["expiry_days"],
        scope=Scope(data["scope"]),
        topic=data.get("topic", ""),
        version=data.get("version", 1),
    )
