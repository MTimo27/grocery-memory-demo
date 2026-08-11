from __future__ import annotations

from dataclasses import replace
from datetime import date

from grocery_memory_demo.memory import evidence_dates, expired, is_pinned, reliability
from grocery_memory_demo.models import Arm, MemoryItem, Metrics, Order, Verdict
from grocery_memory_demo.policy import decide

TRAIN_WEEKS = 12
BASELINE_SUPPORT = 0.3
CLAIM_PRODUCT_MIN_SHARE_GAP = 0.5
CLAIM_HOLDS_SHARE = 0.5


def replay(
    memory: list[MemoryItem], history: list[Order], train_weeks: int = TRAIN_WEEKS
) -> dict[Arm, Metrics]:
    train = [order for order in history if order.week <= train_weeks]
    products_by_topic = {item.topic: claim_products(item, train) for item in memory}
    results = {arm: Metrics() for arm in Arm}
    for test_order in [order for order in history if order.week > train_weeks]:
        known = [order for order in history if order.date < test_order.date]
        for item in memory:
            as_of = rescored(item, known, test_order.date)
            if as_of is None:
                continue
            for arm in Arm:
                arm_verdict = verdict(arm, as_of, known, test_order.date)
                products = products_by_topic[item.topic]
                _record(results[arm], as_of, arm_verdict, products, test_order)
    return results


def verdict(arm: Arm, item: MemoryItem, known: list[Order], today: date) -> Verdict:
    if arm is Arm.HISTORY_ONLY:
        return Verdict.USE if _count_support(item, known) >= BASELINE_SUPPORT else Verdict.IGNORE
    if arm is Arm.WITH_MEMORY:
        return Verdict.IGNORE if expired(item, today) else Verdict.USE
    return decide(item, today)


def rescored(item: MemoryItem, known: list[Order], today: date) -> MemoryItem | None:
    known_ids = {order.id for order in known}
    refs = [ref for ref in item.evidence_refs if ref in known_ids]
    if not refs:
        return None
    as_of = replace(item, evidence_refs=refs, last_evidence=max(evidence_dates(refs, known)))
    return replace(as_of, reliability=reliability(as_of, known, today))


def claim_products(item: MemoryItem, train: list[Order]) -> set[str]:
    if is_pinned(item.status, item.category):
        return set()
    evidence = [order for order in train if order.id in item.evidence_refs]
    background = [order for order in train if order.id not in item.evidence_refs]
    products = {
        product_id
        for product_id in {entry.product_id for order in evidence for entry in order.items}
        if _share(product_id, evidence) - _share(product_id, background)
        >= CLAIM_PRODUCT_MIN_SHARE_GAP
    }
    if not products:
        raise ValueError(f"no distinctive products for claim {item.claim!r}")
    return products


def format_metrics(results: dict[Arm, Metrics], test_weeks: int) -> str:
    header = f"{'arm':<24}{'correct':>9}{'stale':>7}{'missed':>8}{'asked':>7}{'unsafe':>8}"
    rows = [f"Replay over {test_weeks} held-out weeks", header, "-" * len(header)]
    for arm, metrics in results.items():
        rows.append(
            f"{arm.value + '  ' + ARM_LABELS[arm]:<24}"
            f"{metrics.correct:>9}{metrics.stale_errors:>7}"
            f"{metrics.missed:>8}{metrics.clarifications:>7}{metrics.constraint_violations:>8}"
        )
    return "\n".join(rows)


ARM_LABELS = {
    Arm.HISTORY_ONLY: "purchase counts",
    Arm.WITH_MEMORY: "memory",
    Arm.WITH_VERDICTS: "memory + policy",
}


def _record(
    metrics: Metrics,
    item: MemoryItem,
    arm_verdict: Verdict,
    products: set[str],
    test_order: Order,
) -> None:
    if is_pinned(item.status, item.category):
        metrics.constraint_violations += arm_verdict is not Verdict.USE
        return
    if arm_verdict is Verdict.ASK:
        metrics.clarifications += 1
        return
    still_bought = _holds(products, test_order)
    if arm_verdict is Verdict.USE:
        metrics.correct += still_bought
        metrics.stale_errors += not still_bought
    else:
        metrics.missed += still_bought


def _holds(products: set[str], order: Order) -> bool:
    bought = products & {entry.product_id for entry in order.items}
    return len(bought) / len(products) >= CLAIM_HOLDS_SHARE


def _count_support(item: MemoryItem, known: list[Order]) -> float:
    return len(item.evidence_refs) / len(known)


def _share(product_id: str, orders: list[Order]) -> float:
    if not orders:
        return 0.0
    containing = [order for order in orders if _contains(order, product_id)]
    return len(containing) / len(orders)


def _contains(order: Order, product_id: str) -> bool:
    return any(entry.product_id == product_id for entry in order.items)
