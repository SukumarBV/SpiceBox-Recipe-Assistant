from core.ingredient_matcher import match_recipes
from tests.fixtures import make_index, id_of

IDX = make_index()


def test_hard_exclusion_enforced_before_ranking():
    r = match_recipes(IDX, [{"name": "garlic"}, {"name": "noodles"}], ["peanut"], top_n=10)
    ids = [x["id"] for x in r["eligible"]]
    assert id_of(IDX, "Peanut Noodles") not in ids
    assert r["excluded_by_exclusion"] >= 1


def test_no_match_returns_labelled_ineligible_alternatives():
    r = match_recipes(IDX, [{"name": "noodles"}], ["peanut"])
    assert r["eligible"] == [] and r["n_eligible"] == 0
    assert r["alternatives"] and all(a["violations"] for a in r["alternatives"])
    assert any("NO ELIGIBLE" in n for n in r["notes"])


def test_step_limit_and_variant_matching():
    r = match_recipes(IDX, [{"name": "tomatoes"}, {"name": "onion"}], [], max_steps=2, top_n=10)
    assert all(x["steps"] <= 2 for x in r["eligible"])
    salad = [x for x in r["eligible"] if x["title"] == "Quick Tomato Salad"][0]
    assert any(m["match"] == "variant" for m in salad["matched"])  # red onion ~ onion
    assert "quantity sufficiency" in r["notes"][0]


def test_score_is_deterministic_and_explainable():
    a = match_recipes(IDX, [{"name": "eggs"}, {"name": "tomatoes"}, {"name": "rice"}, {"name": "onions"}], ["peanut"], max_steps=6)
    b = match_recipes(IDX, [{"name": "eggs"}, {"name": "tomatoes"}, {"name": "rice"}, {"name": "onions"}], ["peanut"], max_steps=6)
    assert a["eligible"] == b["eligible"]
    top = a["eligible"][0]
    assert top["title"] == "Egg and Tomato Rice Bowl" and top["missing"] == [] and top["n_used"] == 4


# ---------------- scoring audit (v1 bug: 25% usage + 100% coverage + 1 step scored 77.5 and outranked richer matches)
from core.ingredient_matcher import combine, weights_for


def test_old_v1_formula_reproduced_and_new_ranking_prefers_more_ingredients():
    assert round(100 * (0.3 * 0.25 + 0.7 * 1.0), 1) == 77.5      # the audited v1 number
    avail = [{"name": n} for n in ["eggs", "onions", "tomatoes", "rice"]]
    for goal in ("balanced", "most_used", "most_used+fewest_missing", "fewest_missing"):
        r = match_recipes(IDX, avail, ["peanut"], 6, goal, False, top_n=5)
        titles = [x["title"] for x in r["eligible"]]
        assert titles[0] == "Egg and Tomato Rice Bowl"                       # uses all four
        assert titles.index("Plain Rice") > titles.index("Tomato Rice") if "Plain Rice" in titles else True
        used = [x["n_used"] for x in r["eligible"]]
        assert used == sorted(used, reverse=True) or goal == "fewest_missing"


def test_displayed_score_always_reproducible_from_displayed_components():
    for goal in ("balanced", "most_used", "fewest_missing", "closest", "most_used+fewest_missing"):
        for fewer in (False, True):
            r = match_recipes(IDX, [{"name": "rice"}, {"name": "tomatoes"}, {"name": "onion"}], [], None, goal, fewer, top_n=20)
            w = weights_for(goal, fewer)
            assert abs(sum(w) - 1.0) < 0.002
            for x in r["eligible"]:
                assert x["score"] == combine(x["components"], w)


def test_relevance_gate_puts_one_ingredient_recipes_last():
    r = match_recipes(IDX, [{"name": "eggs"}, {"name": "tomatoes"}, {"name": "rice"}, {"name": "onions"}], [], None, "fewest_missing", False, top_n=20)
    flags = [x["partial_only"] for x in r["eligible"]]
    assert flags == sorted(flags)                      # all non-partial first
    assert any(flags)                                  # e.g. Plain Rice uses only 1


def test_peanut_groundnut_and_coconut_milk_distinctions():
    from core.ingredient_normalizer import get_matcher
    assert get_matcher("peanut").search("2 tbsp groundnut oil")
    assert get_matcher("dairy").search("1 cup coconut milk") is None   # coconut milk is not dairy milk
    assert get_matcher("milk").search("1 cup coconut milk") is None
    assert get_matcher("dairy").search("1 cup whole milk")
