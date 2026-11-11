"""LLM layer tests use a FAKE client (no network, no key). They verify validation, fallback and token-economy behaviour."""
import json
from core.llm import LLMClient, LLMConfig, LLMError, LLMRateLimited, interpret, load_config, parse_json
from core.conversation import ConversationState, handle_message
from core import substitution, sub_reasoning
from tests.fixtures import make_index, id_of

IDX = make_index()


class FakeLLM:
    """Mimics LLMClient.complete_json with canned output; counts calls."""
    available = True
    configured = True
    mode = "assist"
    calls = 0

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.sent = []

    def complete_json(self, system, user, max_tokens=700):
        self.calls += 1
        self.sent.append(user)
        o = self.outputs.pop(0)
        if isinstance(o, Exception):
            raise o
        return o

    def status(self):
        return "fake"


def _res(restr=("egg",)):
    rid = id_of(IDX, "Banana Bread")
    return substitution.propose_substitution(IDX, rid, ["egg"], list(restr)), list(restr)


def test_no_key_means_unavailable_and_no_call():
    c = LLMClient(LLMConfig())
    assert not c.available and c.status().startswith("AI off")
    try:
        c.complete_json("s", "u")
        assert False
    except LLMError:
        pass
    assert c.calls == 0


def test_config_from_env_and_secrets(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert load_config({}).provider == ""
    assert load_config({"GEMINI_API_KEY": "k"}).provider == "gemini"
    monkeypatch.setenv("GEMINI_API_KEY", "envkey")
    assert load_config(None).api_key == "envkey"


def test_cache_and_ratelimit_cooldown():
    calls = []

    def transport(url, headers, body):
        calls.append(1)
        assert "key" not in url                       # key travels in a header, never in the URL
        return {"candidates": [{"content": {"parts": [{"text": '{"a": 1}'}]}}]}

    c = LLMClient(LLMConfig("gemini", "m", "KEY"), transport)
    assert c.complete_json("s", "u") == {"a": 1} and c.complete_json("s", "u") == {"a": 1}
    assert len(calls) == 1 and c.cache_hits == 1

    def limited(url, headers, body):
        raise LLMRateLimited("busy")

    c2 = LLMClient(LLMConfig("gemini", "m", "KEY"), limited)
    try:
        c2.complete_json("s", "u2")
    except LLMRateLimited:
        pass
    assert not c2.available                            # cool-down: no retry storm
    assert parse_json('```json\n{"x": 2}\n```') == {"x": 2}


def test_substitution_llm_output_is_validated_and_ids_checked():
    res, restr = _res()
    ev_ids = sub_reasoning.build_prompt(IDX, res, res["slots"][0], restr)[1]
    assert ev_ids and all(i in IDX.id_to_idx for i in ev_ids)
    good = {"role": "binder and moisture", "replacement": "applesauce", "replacement_kind": "candidate", "why": "w",
            "cited_ids": ev_ids[:1] + ["rb_fakefakefake"],
            "ingredients": ["3 bananas", "1 cup sugar", "1/2 cup applesauce", "1 1/2 cups flour"], "steps": ["Mix.", "Bake."],
            "effects": {"flavour": "f", "texture": "t", "preparation": "p"}, "uncertainties": ["u"]}
    fake = FakeLLM([good])
    out = sub_reasoning.refine(fake, IDX, res, restr, "egg")
    assert out["llm"]["used"] and fake.calls == 1
    assert "rb_fakefakefake" not in out["llm"]["cited_ids"]            # hallucinated ID discarded
    assert any("discarded" in n for n in out["llm"]["notes"])
    assert out["llm"]["evidence_class"] in ("Direct", "Weak or conflicting", "Indirect")   # computed by code, not the model
    assert len(fake.sent[0]) < 6000                                    # compact prompt


def test_llm_replacement_violating_restriction_is_rejected():
    res, restr = _res()
    bad = {"role": "r", "replacement": "mayonnaise", "replacement_kind": "general_knowledge", "why": "w", "cited_ids": [],
           "ingredients": ["1 cup mayonnaise"], "steps": ["Bake."], "effects": {}, "uncertainties": []}
    out = sub_reasoning.refine(FakeLLM([bad]), IDX, res, restr, "egg")
    assert not out["llm"]["used"] and "violates" in out["llm"]["reason"]
    assert out["revised_ingredients"] and out["revision_supported"]    # deterministic result kept


def test_llm_failure_and_garbage_fall_back():
    res, restr = _res()
    assert not sub_reasoning.refine(FakeLLM([LLMRateLimited("429")]), IDX, res, restr, "egg")["llm"]["used"]
    res, restr = _res()
    assert not sub_reasoning.refine(FakeLLM([{"replacement": 5}]), IDX, res, restr, "egg")["llm"]["used"]


def test_no_evidence_means_no_llm_call():
    rid = id_of(IDX, "Plain Rice")
    r = substitution.propose_substitution(IDX, rid, ["salt"], ["salt"])
    fake = FakeLLM([])
    sub_reasoning.refine(fake, IDX, r, ["salt"], "salt")
    assert fake.calls == 0


def test_interpretation_only_when_rules_unsure_and_state_stays_source_of_truth():
    st = ConversationState()
    fake = FakeLLM([{"intent": "search", "dish": "rice", "include_ingredients": ["tomato"], "exclude_ingredients": ["peanut"],
                     "available_ingredients": [], "remove_exclusions": [], "max_steps": 6, "top_n": 3, "preference": None, "quick": True,
                     "stat": {}, "substitute": {}}])
    # deterministic commands (reset, restriction management, recipe lookups) never call the LLM
    handle_message("reset", st, IDX, fake)
    assert fake.calls == 0
    # open-ended search requests are interpreted by the LLM first (meaning, not literal words); retrieval stays deterministic
    r = handle_message("zzzz qqqq something yummy", st, IDX, fake)
    assert fake.calls == 1 and "interpreted by AI" in r.text
    assert st.exclusions == ["peanut"] and r.blocks[0]["type"] == "recipes"
    assert all(i["steps"] <= 6 for i in r.blocks[0]["items"])


def test_works_with_llm_disabled_and_unsupported_attributes_refused():
    st = ConversationState()
    r = handle_message("something high-protein under 500 kcal", st, IDX, None)
    assert "no nutrition data" in r.text.lower() and not r.blocks
    r = handle_message("Find rice recipe, ready in 30 minutes", st, IDX, None)
    assert "Not applied" in r.text and r.blocks


def test_general_substitution_without_recipe_is_indirect():
    st = ConversationState()
    r = handle_message("What can I substitute for butter?", st, IDX, None)
    b = r.blocks[0]
    assert b["type"] == "general_sub" and b["result"]["evidence_class"] in ("Indirect", "Insufficient")
