from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

LAST_ORDER_DATE = date(2026, 8, 3)
WEEKS = 16
OUTPUT_PATH = Path(__file__).with_name("history.json")

STAPLES = ["bread_whole_grain", "bananas", "eggs", "chicken_breast", "tomatoes", "coffee_beans"]
ROTATION = [
    ["pasta_penne", "pasta_sauce"],
    ["rice_basmati", "spinach"],
    ["minced_beef", "cucumber"],
    ["salmon_fillet", "apples"],
]
REGULAR_MILK_WEEKS = {1, 2, 6}
KETO_WEEKS = range(5, 10)
PARTY_WEEK = 12
ALLERGY_WEEK = 3
CARBS_DROPPED_DURING_KETO = ["bread_whole_grain", "pasta_penne", "rice_basmati"]


def order_date(week: int) -> date:
    return LAST_ORDER_DATE - timedelta(weeks=WEEKS - week)


def add_staples(basket: dict[str, int], week: int) -> None:
    for product_id in STAPLES:
        basket[product_id] = 1
    for product_id in ROTATION[(week - 1) % len(ROTATION)]:
        basket[product_id] = 1


def add_dairy_preference(basket: dict[str, int], week: int) -> None:
    basket["milk_regular" if week in REGULAR_MILK_WEEKS else "milk_lactose_free"] = 2
    if week % 2 == 0:
        basket["yoghurt_lactose_free" if week >= ALLERGY_WEEK else "yoghurt_greek"] = 1
    if week % 4 == 3:
        basket["cheese_lactose_free" if week >= ALLERGY_WEEK else "cheese_gouda"] = 1


def add_keto_phase(basket: dict[str, int], week: int) -> None:
    if week not in KETO_WEEKS:
        return
    basket["keto_bread"] = 1
    basket["cauliflower_rice"] = 1
    basket["avocado"] = 1
    if week % 2 == 1:
        basket["keto_bar"] = 2
    for product_id in CARBS_DROPPED_DURING_KETO:
        basket.pop(product_id, None)


def add_party(basket: dict[str, int], week: int) -> None:
    if week != PARTY_WEEK:
        return
    basket["chips_paprika"] = 6
    basket["chips_salted"] = 3
    basket["beer_pils"] = 2
    basket["cola"] = 3


def note_for(week: int) -> str | None:
    if week == ALLERGY_WEEK:
        return "Please never send anything with peanuts - my son is allergic."
    if week == PARTY_WEEK:
        return "Having friends over on Saturday."
    return None


def build_order(week: int) -> dict:
    basket: dict[str, int] = {}
    add_staples(basket, week)
    add_dairy_preference(basket, week)
    add_keto_phase(basket, week)
    add_party(basket, week)

    order = {
        "id": f"order_{week:02d}",
        "week": week,
        "date": order_date(week).isoformat(),
        "items": [{"product_id": pid, "qty": qty} for pid, qty in basket.items()],
    }
    note = note_for(week)
    if note:
        order["note"] = note
    return order


def main() -> None:
    history = {
        "household_id": "hh_001",
        "orders": [build_order(week) for week in range(1, WEEKS + 1)],
    }
    OUTPUT_PATH.write_text(json.dumps(history, indent=2) + "\n")
    print(f"{WEEKS} orders -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
