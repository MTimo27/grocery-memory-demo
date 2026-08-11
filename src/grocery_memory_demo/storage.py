from __future__ import annotations

import json
from pathlib import Path

from grocery_memory_demo.models import (
    MemoryItem,
    Order,
    Product,
    memory_item_from_dict,
    order_from_dict,
    product_from_dict,
    to_dict,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
CATALOGUE_PATH = DATA_DIR / "catalogue.json"
HISTORY_PATH = DATA_DIR / "history.json"
MEMORY_PATH = OUT_DIR / "memory.json"


def load_catalogue(path: Path = CATALOGUE_PATH) -> list[Product]:
    return [product_from_dict(entry) for entry in json.loads(path.read_text())]


def load_history(path: Path = HISTORY_PATH) -> list[Order]:
    return [order_from_dict(entry) for entry in json.loads(path.read_text())["orders"]]


def load_memory(path: Path = MEMORY_PATH) -> list[MemoryItem]:
    return [memory_item_from_dict(entry) for entry in json.loads(path.read_text())]


def save_memory(items: list[MemoryItem], path: Path = MEMORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([to_dict(item) for item in items], indent=2) + "\n")
