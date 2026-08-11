from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest

import build_memory
import demo
from grocery_memory_demo import agent, checks, evaluate, loading, memory, policy, storage
from grocery_memory_demo.models import Arm, OrderItem, Scope, Status, Transcript, Verdict, to_dict

HISTORY = storage.load_history()
CATALOGUE = storage.load_catalogue()
MEMORY = storage.load_memory()
TODAY = max(order.date for order in HISTORY)
NEXT_YEAR = TODAY.replace(year=TODAY.year + 1)

PLANTED_CLAIMS: list[dict] = [
    {
        "claim": "Prefers lactose-free dairy (milk, yoghurt, cheese)",
        "category": "dietary_pref",
        "status": "inferred",
        "topic": "lactose_free_dairy",
        "scope": "household",
        "evidence_refs": [f"order_{week:02d}" for week in range(1, 17) if week not in (1, 2, 6)],
    },
    {
        "claim": "Never send products containing peanuts - son is allergic",
        "category": "hard_constraint",
        "status": "explicit",
        "topic": "peanut_allergy",
        "scope": "member",
        "evidence_refs": ["order_03"],
    },
    {
        "claim": "Follows a keto diet: keto bread, cauliflower rice, no bread or pasta",
        "category": "dietary_pref",
        "status": "inferred",
        "topic": "keto_diet",
        "scope": "household",
        "evidence_refs": [f"order_{week:02d}" for week in range(5, 10)],
    },
    {
        "claim": "Buys party snacks and beer in bulk",
        "category": "brand_taste",
        "status": "explicit",
        "topic": "party_snacks",
        "scope": "occasion",
        "evidence_refs": ["order_12"],
    },
]


def claim_for(topic: str) -> dict:
    return next(claim for claim in PLANTED_CLAIMS if claim["topic"] == topic)


def item_for(topic: str):
    return memory.build_item(claim_for(topic), HISTORY, TODAY)


def test_explicit_hard_constraints_pin_to_full_reliability_and_never_expire():
    allergy = item_for("peanut_allergy")
    assert allergy.status is Status.EXPLICIT
    assert allergy.reliability == memory.PINNED_RELIABILITY
    assert allergy.expiry_days is None
    assert not memory.expired(allergy, NEXT_YEAR)


def test_stating_a_one_off_out_loud_does_not_pin_it():
    party = item_for("party_snacks")
    assert party.status is Status.EXPLICIT
    assert party.reliability < memory.PINNED_RELIABILITY
    assert party.expiry_days == memory.INFERRED_EXPIRY_DAYS
    assert memory.expired(party, NEXT_YEAR)


def test_frequent_recent_evidence_outranks_rare_stale_evidence():
    assert item_for("lactose_free_dairy").reliability > item_for("keto_diet").reliability
    assert item_for("keto_diet").reliability > item_for("party_snacks").reliability


def test_reliability_lands_in_the_bands_the_demo_depends_on():
    assert item_for("lactose_free_dairy").reliability > 0.7
    assert 0.3 < item_for("keto_diet").reliability < 0.7
    assert item_for("party_snacks").reliability < 0.3


def test_same_evidence_scores_lower_as_it_ages():
    keto = item_for("keto_diet")
    assert memory.reliability(keto, HISTORY, NEXT_YEAR) < keto.reliability


def test_inferred_claims_expire_once_evidence_outlives_the_window():
    keto = item_for("keto_diet")
    assert not memory.expired(keto, TODAY)
    assert memory.expired(keto, NEXT_YEAR)


def test_rebuilding_from_unchanged_claims_leaves_the_file_identical():
    first = memory.update([], PLANTED_CLAIMS, HISTORY, TODAY)
    second = memory.update(first, PLANTED_CLAIMS, HISTORY, TODAY)
    assert second == first
    assert [item.version for item in second] == [1] * len(first)


def test_revising_a_claim_bumps_its_version_and_drops_absent_claims():
    original = memory.update([], PLANTED_CLAIMS, HISTORY, TODAY)
    revised = [dict(claim_for("lactose_free_dairy"), claim="Prefers lactose-free milk only")]
    result = memory.update(original, revised, HISTORY, TODAY)
    assert [item.version for item in result] == [2]


def test_changing_a_claims_scope_bumps_its_version():
    original = memory.update([], PLANTED_CLAIMS, HISTORY, TODAY)
    revised = [dict(claim_for("lactose_free_dairy"), scope=Scope.MEMBER.value)]
    result = memory.update(original, revised, HISTORY, TODAY)
    assert [item.version for item in result] == [2]


def test_duplicate_claim_identities_are_rejected():
    duplicate = dict(claim_for("keto_diet"), claim="Also follows keto")
    with pytest.raises(ValueError, match="duplicate claim identities"):
        memory.update([], [claim_for("keto_diet"), duplicate], HISTORY, TODAY)


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError):
        memory.build_item(dict(claim_for("keto_diet"), category="vibes"), HISTORY, TODAY)


def test_claims_citing_unknown_orders_are_rejected():
    with pytest.raises(ValueError):
        memory.build_item(dict(claim_for("keto_diet"), evidence_refs=["order_99"]), HISTORY, TODAY)


def test_the_allergy_is_used_however_old_it_gets():
    assert policy.decide(item_for("peanut_allergy"), NEXT_YEAR) is Verdict.USE


def test_the_stable_preference_is_used_and_the_one_off_is_ignored():
    assert policy.decide(item_for("lactose_free_dairy"), TODAY) is Verdict.USE
    assert policy.decide(item_for("party_snacks"), TODAY) is Verdict.IGNORE


def test_the_stale_keto_phase_is_worth_asking_about():
    assert policy.decide(item_for("keto_diet"), TODAY) is Verdict.ASK


def test_expiry_beats_reliability():
    assert policy.decide(item_for("lactose_free_dairy"), NEXT_YEAR) is Verdict.IGNORE


def test_cheap_mistakes_are_never_worth_a_question():
    cheap = policy.MISTAKE_COSTS["brand_taste"]
    reliabilities = [step / 100 for step in range(101)]
    assert all(min(1 - p, p) * cheap <= policy.ASK_COST for p in reliabilities)


def test_the_committed_memory_produces_the_expected_key_verdicts():
    by_topic = policy.verdicts(MEMORY, TODAY)
    assert by_topic["peanut_allergy"] is Verdict.USE
    assert by_topic["lactose_free_dairy"] is Verdict.USE
    assert by_topic["keto_diet"] is Verdict.ASK


def test_a_week_12_occasion_has_no_counterevidence_at_the_initial_cutoff():
    by_topic = policy.verdicts(MEMORY, TODAY)
    assert by_topic["entertaining_occasion"] is Verdict.USE


def test_the_committed_memory_contains_only_training_evidence():
    training_history = [order for order in HISTORY if order.week <= evaluate.TRAIN_WEEKS]
    training_ids = {order.id for order in training_history}
    training_end = max(order.date for order in training_history)

    for item in MEMORY:
        assert set(item.evidence_refs) <= training_ids
        assert item.last_evidence <= training_end


def test_a_peanut_product_violates_the_stated_allergy():
    basket = [OrderItem("peanut_butter", 1), OrderItem("milk_lactose_free", 1)]
    violations = checks.hard_constraint_violations(basket, MEMORY, CATALOGUE)
    assert [violation.product_id for violation in violations] == ["peanut_butter"]
    assert violations[0].allergen == "peanut"


def test_a_safe_basket_passes():
    basket = [OrderItem("milk_lactose_free", 1), OrderItem("bread_whole_grain", 1)]
    assert checks.hard_constraint_violations(basket, MEMORY, CATALOGUE) == []


def test_only_stated_allergens_are_forbidden():
    forbidden = checks.forbidden_allergens(MEMORY, CATALOGUE)
    assert set(forbidden) == {"peanut"}


def test_allergen_matching_accepts_plurals_without_matching_longer_words():
    assert checks._mentions("Never send eggs", "egg")
    assert checks._mentions("Allergic to tree nuts", "tree_nut")
    assert not checks._mentions("Never send eggplant", "egg")


def test_a_basket_of_unknown_products_is_rejected():
    with pytest.raises(ValueError):
        checks.hard_constraint_violations([OrderItem("caviar", 1)], MEMORY, CATALOGUE)


def prompt_for(arm: Arm) -> str:
    return agent.system_prompt(arm, MEMORY, HISTORY, TODAY)


def test_the_arms_differ_only_in_what_they_are_told_about_memory():
    history_only, with_memory, with_verdicts = (prompt_for(arm) for arm in Arm)
    assert "lactose-free" not in history_only
    assert "lactose-free" in with_memory
    assert "-> use" not in with_memory
    assert "-> use" in with_verdicts and "-> ask" in with_verdicts
    assert with_memory.startswith(history_only)


def test_the_agent_only_carts_products_that_exist_and_are_in_stock():
    transcript = Transcript(arm=Arm.WITH_VERDICTS, request="")
    assert "No product" in agent._add_to_cart("caviar", 1, transcript, CATALOGUE)
    assert "out of stock" in agent._add_to_cart("chocolate_dark", 1, transcript, CATALOGUE)
    assert transcript.basket == []
    agent._add_to_cart("milk_lactose_free", 2, transcript, CATALOGUE)
    assert transcript.basket == [OrderItem("milk_lactose_free", 2)]


def test_searching_surfaces_the_variants_the_demo_turns_on():
    results = agent._search_catalogue("milk", CATALOGUE)
    assert "milk_lactose_free" in results and "milk_regular" in results
    assert "No products match" in agent._search_catalogue("caviar", CATALOGUE)


def test_catalogue_search_ignores_short_stopwords_and_substring_matches():
    results = agent._search_catalogue("bag of nuts", CATALOGUE)
    assert "nuts_peanut" in results
    assert "coffee_beans" not in results


def test_loading_memory_rejects_an_unknown_category():
    item = to_dict(MEMORY[0])
    item["category"] = "vibes"
    with pytest.raises(ValueError, match="unknown category"):
        storage.memory_item_from_dict(item)


def test_unknown_demo_scenarios_are_rejected():
    with pytest.raises(SystemExit, match="unknown scenario alergy"):
        demo.selected(["alergy"])


def test_loading_progress_remains_readable_when_output_is_redirected():
    output = StringIO()
    with loading.LoadingScreen(1, output) as screen:
        screen.waiting(1, "1. The weekly shop", Arm.HISTORY_ONLY, 1)
        screen.completed()

    rendered = output.getvalue()
    assert "Loading demo: 0/1 comparisons complete" in rendered
    assert "A purchase counts: waiting for Claude (turn 1)" in rendered
    assert "Demo ready: 1/1 comparisons complete" in rendered


def test_turn_limit_is_reported_in_the_transcript(monkeypatch):
    tool_use = SimpleNamespace(
        type="tool_use",
        name="search_catalogue",
        input={"query": "milk"},
        id="tool-id",
    )
    response = SimpleNamespace(content=[tool_use])
    messages = SimpleNamespace(create=lambda **kwargs: response)
    monkeypatch.setattr(agent, "client", lambda: SimpleNamespace(messages=messages))
    monkeypatch.setattr(agent, "MAX_TURNS", 1)

    turns = []
    transcript = agent.run_arm(
        Arm.HISTORY_ONLY,
        "Add milk",
        "",
        MEMORY,
        CATALOGUE,
        HISTORY,
        TODAY,
        on_model_turn=turns.append,
    )

    assert transcript.reply == "Stopped after reaching the 1-turn limit."
    assert turns == [1]


TRAIN = [order for order in HISTORY if order.week <= evaluate.TRAIN_WEEKS]
TEST_WEEKS = len(HISTORY) - len(TRAIN)
REPLAY = evaluate.replay(MEMORY, HISTORY)


def test_week_13_memory_extraction_cannot_see_current_or_future_orders(monkeypatch):
    # The cutoff is exclusive: the week-13 order is the prediction target, so
    # neither that order nor weeks 14-16 may influence claim discovery, wording,
    # evidence selection, or scoring.
    week_13 = next(order for order in HISTORY if order.week == evaluate.TRAIN_WEEKS + 1)
    extracted_from = []
    scored_from = []
    scored_as_of = []

    def capture_extraction_history(history):
        extracted_from.extend(history)
        return []

    def capture_scoring_history(_memory, _claims, history, today):
        scored_from.extend(history)
        scored_as_of.append(today)
        return []

    monkeypatch.setattr(build_memory.storage, "load_history", lambda: HISTORY)
    monkeypatch.setattr(build_memory.storage, "load_memory", lambda: [])
    monkeypatch.setattr(build_memory.storage, "save_memory", lambda _items: None)
    monkeypatch.setattr(build_memory, "extract_claims", capture_extraction_history)
    monkeypatch.setattr(build_memory, "update", capture_scoring_history)

    build_memory.main()

    expected_history = [order for order in HISTORY if order.date < week_13.date]
    assert extracted_from == expected_history
    assert scored_from == expected_history
    assert scored_as_of == [max(order.date for order in expected_history)]
    assert {order.week for order in extracted_from} == set(range(1, 13))


def products_for(topic: str) -> set[str]:
    item = next(entry for entry in MEMORY if entry.topic == topic)
    return evaluate.claim_products(item, TRAIN)


def test_a_claim_is_replayed_against_the_products_that_are_distinctive_to_it():
    assert products_for("lactose_free_dairy") == {"milk_lactose_free"}
    assert products_for("keto_diet") == {
        "avocado",
        "cauliflower_rice",
        "keto_bar",
        "keto_bread",
    }
    assert "coffee_beans" not in products_for("keto_diet")


def test_hard_constraints_are_replayed_as_the_gate_they_are_not_as_a_pattern():
    assert products_for("peanut_allergy") == set()


def test_only_the_counts_baseline_misses_the_allergy_every_week():
    assert REPLAY[Arm.HISTORY_ONLY].constraint_violations == TEST_WEEKS
    assert REPLAY[Arm.WITH_MEMORY].constraint_violations == 0
    assert REPLAY[Arm.WITH_VERDICTS].constraint_violations == 0


def test_the_policy_arm_trades_stale_personalisation_for_questions():
    counts, with_memory, with_verdicts = (REPLAY[arm] for arm in Arm)
    assert with_verdicts.stale_errors < counts.stale_errors < with_memory.stale_errors
    assert with_verdicts.clarifications > 0
    assert counts.clarifications == with_memory.clarifications == 0


def test_memory_without_a_policy_acts_on_everything_it_remembers():
    with_memory = REPLAY[Arm.WITH_MEMORY]
    assert with_memory.missed == 0
    assert with_memory.stale_errors == max(REPLAY[arm].stale_errors for arm in Arm)
