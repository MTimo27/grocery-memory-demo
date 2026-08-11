from __future__ import annotations

from grocery_memory_demo import storage
from grocery_memory_demo.evaluate import TRAIN_WEEKS
from grocery_memory_demo.extraction import extract_claims
from grocery_memory_demo.memory import update
from grocery_memory_demo.models import MemoryItem


def main() -> None:
    full_history = storage.load_history()
    visible_history = [order for order in full_history if order.week <= TRAIN_WEEKS]

    today = max(order.date for order in visible_history)
    existing = storage.load_memory() if storage.MEMORY_PATH.exists() else []

    claims = extract_claims(visible_history)
    items = update(existing, claims, visible_history, today)
    storage.save_memory(items)

    print(f"{len(items)} claims -> {storage.MEMORY_PATH}  (today={today})")
    for item in items:
        print(_summary_line(item))


def _summary_line(item: MemoryItem) -> str:
    return f"  {item.reliability:>5.3f}  {item.category:<15} v{item.version}  {item.claim}"


if __name__ == "__main__":
    main()
