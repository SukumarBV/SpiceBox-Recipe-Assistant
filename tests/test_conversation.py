from core.conversation import ConversationState, extract_exclusions, handle_message, parse_max_steps
from tests.fixtures import make_index, id_of

IDX = make_index()


def titles(resp, key="items"):
    b = resp.blocks[0]
    if b["type"] == "recipes":
        return [i["title"] for i in b["items"]]
    return [r["title"] for r in b["result"]["eligible"]]


def test_search_then_dairy_followup_preserves_search():
    st = ConversationState()
    r1 = handle_message("Find a penne pasta recipe", st, IDX)
    assert "Penne with Tomato Cream Sauce" in titles(r1) and len(titles(r1)) == 3
    r2 = handle_message("I cannot have dairy", st, IDX)
    assert st.exclusions == ["dairy"] and st.filters["dish_tokens"] == ["penne", "pasta"]
    assert "Penne with Tomato Cream Sauce" not in titles(r2) and len(titles(r2)) == 2


def test_match_followup_exclude_eggs():
    st = ConversationState()
    msg = ("I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, with no peanuts "
           "and at most six instruction steps. Prefer recipes that need the fewest additional ingredients.")
    r1 = handle_message(msg, st, IDX)
    assert [a["name"] for a in st.available] == ["egg", "onion", "tomato", "rice"]
    assert st.exclusions == ["peanut"] and st.preferences["max_steps"] == 6 and st.preferences["goal"] == "most_used+fewest_missing"
    assert "Egg and Tomato Rice Bowl" in titles(r1)
    r2 = handle_message("Exclude eggs too", st, IDX)
    assert "egg" in st.exclusions and "peanut" in st.exclusions and st.preferences["max_steps"] == 6
    assert "egg" not in [a["name"] for a in st.available]
    assert "Egg and Tomato Rice Bowl" not in titles(r2) and "Fried Rice" not in titles(r2)


def test_stats_questions_report_denominator():
    st = ConversationState()
    r = handle_message("How many recipes contain both garlic and onion? What percentage of the dataset is that?", st, IDX)
    res = r.blocks[0]["result"]
    assert res["kind"] == "count" and res["denominator"] == 16 and res["terms"] == ["garlic", "onion"]
    r = handle_message("Among recipes containing tomato, which five have the fewest instruction steps?", st, IDX)
    res = r.blocks[0]["result"]
    assert res["kind"] == "rank_steps" and len(res["rows"]) == 5 and res["rows"][0]["title"] == "Quick Tomato Salad"
    assert res["rows"][0]["id"].startswith("rb_")


def test_no_match_message_is_explicit():
    st = ConversationState()
    r = handle_message("I have noodles. No peanuts.", st, IDX)
    assert "No eligible recipe" in r.text and r.blocks[0]["result"]["alternatives"]


def test_substitution_flow_and_reset():
    st = ConversationState()
    r = handle_message("Take the banana bread recipe and remove eggs.", st, IDX)
    res = r.blocks[0]["result"]
    assert res["recipe"]["title"] in ("Banana Bread", "Banana Walnut Bread")
    assert res["slots"][0]["target"] == "egg" and "egg" in st.exclusions
    assert res["recipe"]["ingredients"]  # original preserved separately
    assert res["revision_supported"] and not res["remaining_violations"]
    handle_message("reset", st, IDX)
    assert st.exclusions == [] and st.intent is None


def test_parsers():
    assert extract_exclusions("with no peanuts and at most six instruction steps.") == ["peanut"]
    assert extract_exclusions("I cannot have dairy or eggs") == ["dairy", "egg"]
    assert extract_exclusions("a dairy-free dessert") == ["dairy"]
    assert extract_exclusions("no more than six steps") == []
    assert parse_max_steps("at most six instruction steps") == 6 and parse_max_steps("under 5 steps") == 4
