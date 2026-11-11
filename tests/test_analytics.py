from core import analytics
from tests.fixtures import make_index

IDX = make_index()


def test_count_both_garlic_and_onion():
    r = analytics.count_containing(IDX, ["garlic", "onion"])
    titles = {IDX.titles[IDX.id_to_idx[i]] for i in r["sample_ids"]}
    assert r["denominator"] == len(IDX) == 16
    assert r["count"] == 3  # constructed fixture: Tomato Soup, Tomato Rice, Garlic Onion Chicken
    assert r["percentage"] == round(100 * 3 / 16, 2)
    assert "whole-word" in r["normalization"]


def test_exact_vs_token_and_scope():
    r = analytics.count_containing(IDX, ["tomato"])
    assert r["count"] >= r["exact_name_count"] > 0
    base = {i for i in range(len(IDX)) if "rice" in IDX.title_tokens[i]}
    r2 = analytics.count_containing(IDX, ["tomato"], base, ["title contains: rice"])
    assert r2["denominator"] == len(base) and r2["count"] <= len(base)


def test_rank_by_steps_deterministic_and_ties():
    r = analytics.rank_by_steps(IDX, IDX.contains("tomato"), 3, True)
    steps = [row["steps"] for row in r["rows"]]
    assert steps == sorted(steps) and steps[0] == 2
    assert r["rows"][0]["title"] == "Quick Tomato Salad"
    assert r == analytics.rank_by_steps(IDX, IDX.contains("tomato"), 3, True)


def test_ingredient_stats_and_top():
    s = analytics.ingredient_count_stats(IDX)
    counts = [IDX.ing_counts[i] for i in range(len(IDX))]
    assert s["n_analyzed"] == 16 and s["min"] == min(counts) and s["max"] == max(counts)
    t = analytics.top_ingredients(IDX, None, 3)
    assert t["rows"][0]["ingredient"] == "salt"
