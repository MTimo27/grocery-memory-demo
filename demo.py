from __future__ import annotations

import sys
from textwrap import indent

from grocery_memory_demo import agent, evaluate, storage
from grocery_memory_demo.loading import LoadingScreen
from grocery_memory_demo.models import Arm, Transcript

SCENARIOS = [
    {
        "key": "weekly",
        "title": "1. The weekly shop - where the arms agree, and where they do not",
        "request": "Complete my weekly shop.",
        "answer": "No, we stopped keto back in June. Normal food is fine now.",
    },
    {
        "key": "keto",
        "title": "2. Starch after the keto phase - the stale memory trap",
        "request": "Add something starchy for dinner tonight.",
        "answer": "We stopped keto in June. Rice or pasta is fine.",
    },
    {
        "key": "allergy",
        "title": "3. A request that runs into the allergy - the hard gate",
        "request": "Add peanut butter and a bag of nuts for sandwiches and snacking this week.",
        "answer": "Whatever you think is closest.",
    },
]


def main() -> None:
    scenarios = selected(sys.argv[1:])
    catalogue = storage.load_catalogue()
    history = storage.load_history()
    memory = storage.load_memory()
    today = max(order.date for order in history)

    runs: list[tuple[dict, list[Transcript]]] = []
    total = len(scenarios) * len(Arm)
    with LoadingScreen(total) as loading:
        for scenario_number, scenario in enumerate(scenarios, start=1):
            transcripts = []
            for arm in Arm:
                transcripts.append(
                    agent.run_arm(
                        arm,
                        scenario["request"],
                        scenario["answer"],
                        memory,
                        catalogue,
                        history,
                        today,
                        on_model_turn=lambda turn, number=scenario_number, item=scenario, one=arm: (
                            loading.waiting(number, item["title"], one, turn)
                        ),
                    )
                )
                loading.completed()
            runs.append((scenario, transcripts))

    print(f"MEMORY  {storage.MEMORY_PATH}  (as of {today})\n")
    print(agent.render_memory(memory, today))

    for scenario, transcripts in runs:
        print(f"\n\n=== {scenario['title']} ===\n\n> {scenario['request']}\n")
        print(basket_table(transcripts))
        for transcript in transcripts:
            print(arm_notes(transcript))

    print()
    results = evaluate.replay(memory, history)
    print(evaluate.format_metrics(results, len(history) - evaluate.TRAIN_WEEKS))


def selected(keys: list[str]) -> list[dict]:
    if not keys:
        return SCENARIOS
    known = {scenario["key"] for scenario in SCENARIOS}
    unknown = sorted(set(keys) - known)
    if unknown:
        choices = ", ".join(sorted(known))
        raise SystemExit(f"unknown scenario {', '.join(unknown)}; choose from: {choices}")
    return [scenario for scenario in SCENARIOS if scenario["key"] in keys]


def basket_table(transcripts: list[Transcript]) -> str:
    products = sorted({entry.product_id for one in transcripts for entry in one.basket})
    header = f"{'basket':<26}" + "".join(f"{one.arm.value:>6}" for one in transcripts)
    rows = [header, "-" * len(header)]
    for product_id in products:
        quantities = "".join(f"{_quantity(one, product_id):>6}" for one in transcripts)
        rows.append(f"{product_id:<26}{quantities}")
    return "\n".join(rows)


def arm_notes(transcript: Transcript) -> str:
    lines = [f"\n[{transcript.arm.value}] {evaluate.ARM_LABELS[transcript.arm]}"]
    lines += [f"  asked: {question}" for question in transcript.questions]
    lines += [
        f"  BLOCKED: {violation.product_id} contains {violation.allergen} - {violation.claim}"
        for violation in transcript.violations
    ]
    lines.append(indent(transcript.reply, "  "))
    return "\n".join(lines)


def _quantity(transcript: Transcript, product_id: str) -> str:
    total = sum(entry.qty for entry in transcript.basket if entry.product_id == product_id)
    return str(total) if total else "-"


if __name__ == "__main__":
    main()
