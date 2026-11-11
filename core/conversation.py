"""Rule-based conversation manager: structured session state + intent routing (no LLM, no agent framework)."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

from . import analytics, ingredient_matcher, llm as llm_mod, preferences as prefs_mod, sub_reasoning, substitution
from .ingredient_normalizer import canonical_exclusion, describe_matcher, get_matcher, normalize_ingredient, singularize, split_quantity
from .search import dish_tokens, search_recipes

NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
       "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
_NUMRE = r"\d+|" + "|".join(NUM)


def to_int(tok):
    tok = (tok or "").lower()
    return int(tok) if tok.isdigit() else NUM.get(tok)


@dataclass
class ConversationState:
    intent: str | None = None
    active_recipe_id: str | None = None
    available: list = field(default_factory=list)
    exclusions: list = field(default_factory=list)
    preferences: dict = field(default_factory=lambda: {"goal": "balanced", "prefer_fewer_steps": False, "max_steps": None,
                                                       "top_n": 5, "assume_staples": True})
    filters: dict = field(default_factory=lambda: {"dish_tokens": [], "ingredient_terms": []})
    last_result_ids: list = field(default_factory=list)
    last_substitution: dict | None = None
    last_stat: dict | None = None

    def reset(self):
        self.__init__()

    def to_dict(self):
        return {"intent": self.intent, "active_recipe_id": self.active_recipe_id, "available_ingredients": [a["name"] for a in self.available],
                "hard_exclusions": list(self.exclusions), "preferences": dict(self.preferences),
                "search_filters": dict(self.filters), "previous_result_ids": list(self.last_result_ids)}


@dataclass
class Response:
    text: str
    blocks: list = field(default_factory=list)
    confident: bool = True


# ------------------------------------------------------------------ parsing helpers
EXCL_RE = re.compile(
    r"\b(?:can(?:not|'t|t)|can\s+not|am\s+not\s+able\s+to|don't|do\s+not|must\s+not|mustn't|shouldn't|should\s+not)\s+(?:have|eat|use|consume|tolerate|include)\s+(?P<a>[^.;!?]+)"
    r"|\b(?:exclude|excluding|avoid|avoiding|without|no|allergic\s+to|allergy\s+to|intolerant\s+to|free\s+of|not\s+including|skip|skipping|hold\s+the)\s+(?P<b>[^.;!?]+)", re.I)
FREE_RE = re.compile(r"\b([a-z]+)[- ]free\b", re.I)
STOP_CUT = re.compile(r"(?:,|\b(?:and\s+at\s+most|at\s+most|no\s+more\s+than|at\s+least|with\s+at\s+most|and\s+prefer|prefer|but|please|also|too|as\s+well|and\s+use|using|then|which|that|where|for|so|because|in|from|when|if|and\s+make|and\s+give|and\s+show|and\s+suggest)\b).*$", re.I)
FILLER = {"any", "all", "the", "of", "my", "a", "an", "some", "anything", "containing", "contain", "contains", "with", "food", "foods",
          "products", "product", "item", "items", "ingredients", "ingredient", "recipes", "recipe", "dishes", "dish"}
BAD_WORDS = {"step", "steps", "more", "than", "fewer", "less", "extra", "additional", "problem", "idea", "way", "need", "matter",
             "longer", "time", "one", "feel", "hassle", "cost", "fat", "sugar-free"}


def _clean_items(phrase: str, max_words=3):
    out = []
    for part in re.split(r",|\band\b|\bor\b|\bnor\b|&|/|\bplus\b", phrase.lower()):
        w = [x for x in re.findall(r"[a-z][a-z\-']*", part) if x not in FILLER]
        if not w or len(w) > max_words or any(x in BAD_WORDS for x in w):
            continue
        out.append(" ".join(w))
    return out


def extract_exclusions(text: str) -> list[str]:
    text = text.replace("’", "'")
    found = []
    for m in EXCL_RE.finditer(text):
        phrase = STOP_CUT.sub("", m.group("a") or m.group("b") or "")
        for item in _clean_items(phrase):
            found.append(item)
    for m in FREE_RE.finditer(text):
        if m.group(1).lower() not in BAD_WORDS:
            found.append(m.group(1).lower())
    out = []
    for f in found:
        c = " ".join(singularize(w) for w in re.findall(r"[a-z]+", f))
        c = canonical_exclusion(c) if c else ""
        if c and c not in out:
            out.append(c)
    return out


def strip_exclusions(text: str) -> str:
    text = text.replace("’", "'")
    return FREE_RE.sub(" ", EXCL_RE.sub(" ", text))


STEPS_RE = re.compile(r"\b(?P<kind>at most|no more than|not more than|up to|maximum of|maximum|max|within|under|fewer than|less than|below)\s+(?P<n>\d+|[a-z]+)\s+(?:(?:instruction|recipe|cooking|preparation)\s+)?steps?\b", re.I)
STEPS_RE2 = re.compile(r"\b(?P<n>\d+|[a-z]+)\s+(?:(?:instruction|recipe|cooking)\s+)?steps?\s+or\s+(?:fewer|less)\b", re.I)
TOPN_RES = [re.compile(rf"\b(?:top|which|first|best|suggest|give me|show me|list|find me|recommend)\s+(?P<n>{_NUMRE})\b", re.I),
            re.compile(rf"\b(?P<n>{_NUMRE})\s+(?:\w+\s+){{0,2}}recipes?\b", re.I)]


def parse_max_steps(text):
    m = STEPS_RE.search(text)
    if m:
        n = to_int(m.group("n"))
        if n is not None:
            return max(n - 1, 0) if m.group("kind").lower() in ("under", "fewer than", "less than", "below") else n
    m = STEPS_RE2.search(text)
    if m and to_int(m.group("n")) is not None:
        return to_int(m.group("n"))
    return None


def parse_top_n(text):
    for rx in TOPN_RES:
        m = rx.search(text)
        if m:
            n = to_int(m.group("n"))
            if n and 1 <= n <= 20:
                return n
    return None


def parse_preferences(low: str) -> dict:
    p = {}
    most = bool(re.search(r"as many (of )?(these|my|the|available)?.*?as possible|use (up )?(as many|most)|most of my|more of my", low))
    fewest = bool(re.search(r"(fewest|fewer|least|minimal|minimum|minimi[sz]e)\s+(additional|missing|extra)|need the fewest|few(er|est)? (additional|extra|missing)", low))
    if most and fewest:
        p["goal"] = "most_used+fewest_missing"
    elif fewest:
        p["goal"] = "fewest_missing"
    elif most:
        p["goal"] = "most_used"
    elif re.search(r"closest( recipe)? match|best match|closest recipe", low):
        p["goal"] = "closest"
    if re.search(r"fewest (instruction )?steps|fewer steps|simplest|easiest|quickest|shortest|least steps|\bquick\b|\bfast\b", low):
        p["prefer_fewer_steps"] = True
    if re.search(r"(don't|do not|no|without)\s+(assum\w+\s+)?(pantry\s+)?staples", low):
        p["assume_staples"] = False
    elif re.search(r"assume .*(pantry|staples)|pantry staples", low):
        p["assume_staples"] = True
    return p


# Phrases that look like nutrition words but are handled as transparent ingredient-evidence preferences.
_PREF_OK = re.compile(r"\b(high[- ]protein|protein[- ](?:rich|packed)|lots of protein|more protein|low[- ]carb)\b", re.I)
MORE_RE = re.compile(r"^\s*(?:show\s+(?:me\s+)?)?(?:some\s+)?(?:more|other|different|next)(?:\s+(?:options|results|recipes|ideas|ones|suggestions))?\s*[.!?]*\s*$|^\s*(?:anything|something)\s+else\s*[.!?]*\s*$", re.I)
FOLLOWUP_CUE = re.compile(r"\b(make\s+(?:it|them|that|this)|instead|but|same|those|these|them|too|also|again|rather|actually|now|lighter|spicier|milder|quicker|faster|healthier|heartier|easier|more\s+\w+|less\s+\w+)\b", re.I)

UNSUP_RE = re.compile(r"\b(calories?|kcal|nutrition\w*|protein|carbs?|carbohydrates?|sodium|cholesterol|vitamins?|(?:prep(?:aration)?|cook(?:ing)?|total) time|how long|ready in|\d+\s*(?:minutes?|mins?|hours?)|servings?|serves|per serving|price|cost|cheap|budget|rating|rated|difficulty)\b", re.I)
QUICK_RE = re.compile(r"\b(quick|fast|easy|simple|speedy|weeknight)\b", re.I)


AVAIL_RE = re.compile(r"\b(?:i\s+(?:only\s+)?have|i've\s+got|i\s+got|we\s+have|available\s+ingredients?\s*(?:are|:|include)?|my\s+ingredients\s+(?:are|:)|ingredients?\s+i\s+have\s*(?:are|:)?|what\s+can\s+i\s+(?:make|cook)\s+with|with\s+these\s+ingredients\s*:?)\s*(?P<x>[^.;!?]+)", re.I)
MATCH_CUT = re.compile(r"\b(?:suggest|find|give|show|recommend|list|what|with\s+no|without|and\s+no|no\s|but|please|and\s+at\s+most|at\s+most|and\s+prefer|prefer|i\s+want|i\s+need|using\s+as\s+many)\b.*$", re.I)
ADD_RE = re.compile(r"\b(?:i\s+also\s+have|also\s+have|add|plus|i\s+now\s+have|and\s+i\s+have)\s+(?P<x>[^.;!?]+)", re.I)
REMOVE_AV_RE = re.compile(r"\b(?:remove|drop|delete)\s+(?P<x>[^.;!?]+?)\s+from\s+(?:my\s+)?(?:ingredients|list)|\bi\s+(?:ran\s+out\s+of|don't\s+have|do\s+not\s+have)\s+(?P<y>[^.;!?]+?)(?:\s+(?:anymore|any more))?\s*(?:[.;!?]|$)|\bno\s+more\s+(?P<z>[^.;!?]+)", re.I)


def parse_available(phrase: str) -> list[dict]:
    phrase = MATCH_CUT.sub("", phrase)
    items = []
    for part in re.split(r",|\band\b|&|\bplus\b", phrase):
        part = re.sub(r"^\s*(?:some|a few|a|an|the|my|about|just|also|of)\s+", "", part.strip(), flags=re.I)
        if not part.strip():
            continue
        name = normalize_ingredient(part)
        if not name or len(name.split()) > 4 or any(x in BAD_WORDS for x in name.split()):
            continue
        qty, unit, _ = split_quantity(part)
        if name not in [i["name"] for i in items]:
            items.append({"name": name, "qty": (qty + " " + unit).strip()})
    return items


SUB_CUE = re.compile(r"\b(replace|replacement|substitut\w*|swap\w*|instead of|in place of|take out|get rid of|eliminate|remove|leave out|alternative (?:to|for)|adapt)\b|make (?:it|this|that)\b[^.?!]*\bfree\b", re.I)
SUB_PATTERNS = [
    (re.compile(r"\b(?P<verb>replace|substitute|swap(?:\s+out)?|eliminate|take\s+out|get\s+rid\s+of|remove|leave\s+out|drop)\s+(?:for\s+)?(?:the\s+|all\s+|any\s+|my\s+)?(?P<t>[a-z][a-z\- ]*?)(?=\s+(?:in|from|with|for|using|and|without|by|but|so|because|as|so)\b|[.,;?!]|$)", re.I), None),
    (re.compile(r"\b(?P<verb>instead\s+of|in\s+place\s+of|replacement\s+for|alternatives?\s+(?:to|for)|substitutes?\s+for)\s+(?:the\s+)?(?P<t>[a-z][a-z\- ]*?)(?=\s+(?:in|from|with|for|using|and|but|so)\b|[.,;?!]|$)", re.I), "replace"),
    (re.compile(r"\bmake\s+(?:it|this|that)\b[^.?!]*?\b(?P<t>[a-z]+)[- ]free\b", re.I), "remove"),
]
GENERIC_TITLE = {"this", "the", "that", "my", "a", "an", "same", "previous", "last", "it"}
TITLE_RE = re.compile(r"(?:\b(?:in|from|for|take|use|using|to)\s+)(?:(?:this|the|that|my|a|an)\s+)?(?P<t>[a-z][a-z'\- ]*?)\s+recipe\b", re.I)
ID_RE = re.compile(r"\brb_[0-9a-f]{8,}(?:-\d+)?\b")
REF_NUM = re.compile(r"\b(?:result|option|recipe|number|no\.?)\s*#?\s*(\d+)\b|#(\d+)\b", re.I)
REF_ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b\s*(?:one|result|recipe|option)?", re.I)
DETAIL_KW = re.compile(r"\b(ingredients?|instructions?|directions?|steps|method|how (?:do|to) (?:i )?(?:make|cook)|full recipe|show|display|details?|read|view)\b", re.I)
STATS_CUE = re.compile(r"\b(how many|number of|count|percentage|percent|proportion|average|mean|median|most common|commonly|popular|frequent|fewest|least|most|rank|ranking|top)\b", re.I)

HELP = ("I can (1) **search** recipes and answer dataset questions (e.g. *Find a penne pasta recipe*, then *I cannot have dairy*), "
        "(2) **match** recipes to ingredients you have (*I have eggs, onions and rice … no peanuts, at most six steps*), and "
        "(3) **adapt** a recipe with evidence from the dataset (*Remove eggs from the banana bread recipe*).")


# ------------------------------------------------------------------ main entry
def handle_message(text: str, state: ConversationState, index, llm=None) -> Response:
    """Hybrid routing: deterministic rules first; the optional LLM only INTERPRETS (never answers facts) when rules are unsure
    (mode 'assist') or for every message (mode 'always'). All retrieval/calculation/validation stays in Python/SQLite."""
    t = (text or "").strip().replace("’", "'")
    use_llm = bool(llm and llm.available)
    if use_llm and llm.mode == "always" and t and not re.fullmatch(r"(reset|start over|new conversation)\W*", t.lower()):
        r = _try_llm(t, state, index, llm)
        if r:
            return r
    resp = _handle_rules(t, state, index, llm)
    if use_llm and not resp.confident:
        r = _try_llm(t, state, index, llm)
        if r:
            return r
    return resp


def _try_llm(text, state, index, llm):
    cmd = llm_mod.interpret(llm, text, state.to_dict())
    if not cmd or cmd["intent"] == "chat":
        return None
    try:
        r = _execute_command(cmd, state, index, llm)
    except Exception:
        return None
    if r:
        r.text = "🤖 *Request interpreted by AI; retrieval and checks done by the database.* " + r.text
    return r


def _unsup_categories(low):
    cats = set()
    for m in UNSUP_RE.finditer(low):
        w = m.group(0).lower()
        if re.search(r"calor|kcal|nutri|protein|carb|sodium|cholest|vitamin", w):
            cats.add("nutrition")
        elif re.search(r"time|how long|ready|minute|min|hour", w):
            cats.add("cooking/preparation time")
        elif re.search(r"serv", w):
            cats.add("serving-size")
        elif re.search(r"price|cost|cheap|budget", w):
            cats.add("price")
        else:
            cats.add("rating/difficulty")
    return sorted(cats)


def _unsupported_reply(low):
    terms = _unsup_categories(low)
    return Response("The recipe collection only contains each recipe's **title, ingredient list and instructions**. It has no "
                    + ", ".join(terms) + " data, so I can't answer or filter on that — and I won't estimate it. "
                    "I *can* search by dish or ingredients, apply restrictions, match the ingredients you have, and count or rank recipes by "
                    "instruction steps (a rough complexity proxy, not a time estimate).")


def _handle_rules(text, state, index, llm=None) -> Response:
    low = text.lower()
    if not text:
        return Response(HELP, confident=False)
    if re.fullmatch(r"(reset|start over|new conversation)\W*", low):
        state.reset()
        return Response("Conversation reset. All session constraints were cleared.")
    r = _manage_restrictions(text, low, state)
    if r:
        return r
    low = _PREF_OK.sub(" ", low)
    unsup = UNSUP_RE.search(low)
    if unsup and not AVAIL_RE.search(text):
        residual = [t for t in dish_tokens(UNSUP_RE.sub(" ", strip_exclusions(text))) if t not in {"under", "over", "than", "less", "more", "high", "low", "tell", "number", "which", "fewest", "most", "least", "lowest", "highest", "fewer", "healthiest", "healthy", "what", "how", "much", "many"}]
        refers = bool(ID_RE.search(low) or REF_NUM.search(text) or REF_ORD.search(low) or re.search(r"\b(this|that|it)\b", low))
        if not residual or refers:
            return _unsupported_reply(low)
    resp = _route(text, low, state, index, llm)
    if unsup and resp is not None:
        terms = _unsup_categories(low)
        resp.text += f"\n\n⚠️ Not applied: the dataset has no {', '.join(terms)} information, so that part of your request was ignored (nothing was estimated)."
    return resp


def _route(text, low, state, index, llm):
    has_ref = bool(ID_RE.search(low) or REF_NUM.search(text) or REF_ORD.search(low) or re.search(r"\b(this|that|it)\b", low))
    if state.intent == "search" and MORE_RE.match(text):
        return _more_results(state, index)
    if AVAIL_RE.search(text):
        return _do_match(text, low, state, index, new_list=True)
    if state.intent == "match" and REMOVE_AV_RE.search(text):
        return _do_match(text, low, state, index, new_list=False)
    if SUB_CUE.search(text) and not (state.intent == "match" and re.search(r"ingredients|my list", low)):
        return _do_substitution(text, low, state, index, llm)
    if has_ref and (state.active_recipe_id or state.last_result_ids or ID_RE.search(low)):
        f = _do_recipe_fact(text, low, state, index)
        if f:
            return f
    if (ID_RE.search(low) or REF_NUM.search(text) or REF_ORD.search(low) or re.search(r"\b(this|that|it|the)\s+(recipe|one)\b", low)
            or re.search(r"\b(ingredients?|instructions?|directions?)\b", low)) and DETAIL_KW.search(text) and not STATS_CUE.search(text):
        resp = _do_details(text, low, state, index)
        if resp:
            return resp
    if STATS_CUE.search(text):
        resp = _do_stats(text, low, state, index)
        if resp:
            return resp
    if state.intent == "match":
        ex = extract_exclusions(text)
        if ex or parse_max_steps(text) is not None or ADD_RE.search(text) or parse_preferences(low) or parse_top_n(text):
            return _do_match(text, low, state, index, new_list=False)
    if state.intent == "substitution" and state.last_substitution and extract_exclusions(text):
        return _redo_substitution(text, state, index, llm)
    # Open-ended requests ("something spicy", "light meal for a hot summer") go to the LLM interpreter FIRST so the
    # meaning, not the literal words, drives retrieval. Deterministic lexicon below is the offline fallback.
    if llm is not None and llm.available:
        r = _try_llm(text, state, index, llm)
        if r:
            return r
    return _do_search(text, low, state, index)


def _manage_restrictions(text, low, state):
    if re.search(r"\b(clear|reset|remove|drop|lift|ignore)\b.*\b(restrictions?|exclusions?|constraints?)\b", low):
        gone = [e for e in state.exclusions if e in low]
        if gone:
            state.exclusions = [e for e in state.exclusions if e not in gone]
            msg = f"Removed restriction(s): {', '.join(gone)}."
        else:
            state.exclusions = []
            msg = "Cleared all hard restrictions."
        return Response(msg + " Active restrictions: " + (", ".join(state.exclusions) or "none") + ".")
    m = re.search(r"\bi\s+can\s+(?:eat|have)\s+(?P<x>[^.;!?]+?)(?:\s+(?:again|now))?\s*(?:[.;!?]|$)", low)
    if m:
        items = [canonical_exclusion(i) for i in _clean_items(m.group("x"))]
        gone = [e for e in state.exclusions if e in items]
        if gone:
            state.exclusions = [e for e in state.exclusions if e not in gone]
            return Response(f"Removed restriction(s): {', '.join(gone)}. Active restrictions: {', '.join(state.exclusions) or 'none'}.")
    return None


def _add_exclusions(state, new):
    added = [e for e in new if e not in state.exclusions]
    state.exclusions.extend(added)
    return added


def _restr_text(state):
    return ", ".join(describe_matcher(get_matcher(e)) for e in state.exclusions) or "none"


# ------------------------------------------------------------------ search
def _ingredient_phrase_terms(text):
    terms = []
    for m in re.finditer(r"\b(?:with|containing|contains|that\s+contain|that\s+uses?|using|made\s+with|includes?|including)\s+(?P<x>[^.;!?]+)", text, re.I):
        phrase = STOP_CUT.sub("", m.group("x"))
        terms += _clean_items(phrase)
    out = []
    for t in terms:
        n = normalize_ingredient(t)
        if n and n not in out:
            out.append(n)
    return out


def _prev_criteria(state):
    if state.intent != "search":
        return None
    return prefs_mod.Criteria.from_dict(state.filters.get("criteria"))


def _do_search(text, low, state, index):
    new_ex = extract_exclusions(text)
    stripped = strip_exclusions(text)
    # Offline understanding of descriptive words (used when Gemini is off or declined).
    pc, diets, stripped = prefs_mod.parse_request(stripped)
    for d in diets:
        if d not in new_ex:
            new_ex.append(d)
    base = UNSUP_RE.sub(" ", stripped)
    ing_terms = _ingredient_phrase_terms(base)
    base_wo = re.sub(r"\b(?:with|containing|contains|that\s+contain|that\s+uses?|using|made\s+with|includes?|including)\s+[^.;!?]+", " ", base, flags=re.I)
    toks = [t for t in dish_tokens(base_wo) if t not in {"now", "actually", "only", "too", "cannot", "can", "not", "t", "cant", "eat", "tolerate"}]
    added = _add_exclusions(state, new_ex)
    top_n = parse_top_n(text) or 5
    ms = parse_max_steps(text)
    quick = bool(QUICK_RE.search(text))
    crit = pc if pc.has_signals() else None
    prev_c = _prev_criteria(state)
    prev_has = state.intent == "search" and bool(state.filters.get("dish_tokens") or state.filters.get("ingredient_terms") or (prev_c and prev_c.has_signals()))
    if not toks and not ing_terms:
        if prev_has and ((crit is None and (added or ms is not None)) or (crit is not None and FOLLOWUP_CUE.search(text))):
            toks, ing_terms = state.filters.get("dish_tokens", []), state.filters.get("ingredient_terms", [])
            crit = prefs_mod.merge_criteria(prev_c, pc) if prev_c else crit
        elif crit is None:
            if added:
                return Response(f"Noted hard restriction(s): {', '.join(added)}. Tell me what to search for, list ingredients you have, or name a recipe to adapt.")
            return Response("I couldn't tell what to look for. " + HELP, confident=False)
    if crit is not None:
        crit = prefs_mod.verify_hints(index, crit)
        if quick and "quick" not in crit.tags:
            crit.tags.append("quick")
            crit.quick = True
    return _run_search(state, index, toks, ing_terms, added, top_n, quick, ms, criteria=crit)


_SCOPE_LABELS = {"title": "title", "title_or_ingredients": "title or ingredient list (no title matched all words)",
                 "ingredients": "ingredient list", "preferences": "preference evidence in titles and ingredient lists"}


def _run_search(state, index, toks, ing_terms, added, top_n=5, quick=False, max_steps=None, criteria=None, offset=0):
    if max_steps is None and state.intent == "search":
        max_steps = state.filters.get("max_steps")
    crit = criteria if (criteria is not None and criteria.has_signals()) else None
    state.intent = "search"
    state.filters = {"dish_tokens": toks, "ingredient_terms": ing_terms, "max_steps": max_steps,
                     "criteria": crit.to_dict() if crit else None, "offset": offset, "top_n": top_n, "quick": quick}
    res = search_recipes(index, toks, ing_terms, state.exclusions, limit=top_n, max_steps=max_steps, prefer_short=quick,
                         criteria=crit, offset=offset)
    state.last_result_ids = [i["id"] for i in res["items"]]
    state.active_recipe_id = res["items"][0]["id"] if len(res["items"]) == 1 else None
    desc = " ".join(toks) + (f" containing {', '.join(ing_terms)}" if ing_terms else "") + (f" (≤{max_steps} steps)" if max_steps is not None else "")
    if crit:
        d2 = crit.describe()
        desc = (desc.strip() + " · " if desc.strip() else "") + d2
    note = f" (added restriction: {', '.join(added)})" if added else ""
    if not res["items"]:
        why = ""
        if res["total_before_exclusions"] and state.exclusions:
            why = f" {res['total_before_exclusions']} recipe(s) matched before restrictions, but all violate: {_restr_text(state)}."
        if offset:
            return Response("That was the last of the matching recipes for this search. Ask for something different or loosen a filter.", [], confident=False)
        return Response(f"No recipes found for “{desc.strip()}” with active restrictions [{_restr_text(state)}].{why} "
                        "Nothing was relaxed silently — say *clear restrictions* or change the search to continue.", [], confident=False)
    scope = _SCOPE_LABELS.get(res["scope"], "preference evidence combined with title/ingredient matching")
    count = f"**{res['total']}**" if res.get("total_exact", True) else f"roughly **{res['total']:,}** candidate"
    shown = f"showing {offset + 1}–{offset + len(res['items'])}" if offset else f"showing {len(res['items'])}"
    txt = (f"Found {count} matching recipe(s) for “{desc.strip()}” (matched on {scope}); {shown}{note}"
           f"{', fewest steps first' if quick and not crit else ''}. Active restrictions: {_restr_text(state)}.")
    if res["excluded_count"]:
        txt += f" {res['excluded_count']} otherwise-matching recipe(s) were removed by your restrictions."
    if crit:
        txt += (" Ranked by evidence found in each recipe's **title and ingredient list** — the database has no flavour, cuisine, meal-type, "
                "nutrition or cooking-time labels, so these are transparent heuristics, not verified labels.")
        if crit.quick or "quick" in crit.tags:
            txt += " “Quick” = fewer instruction steps (no cooking times are stored)."
        if crit.cuisine and crit.cuisine not in prefs_mod.CUISINES:
            txt += f" No built-in cuisine heuristic exists for “{crit.cuisine}”, so it was not used as a filter."
        if crit.evidence_terms or crit.title_keywords:
            txt += " AI-suggested keywords used (all verified to exist in the database): " + ", ".join(crit.title_keywords + crit.evidence_terms) + "."
        if res.get("relaxed"):
            txt += " ⚠️ Relaxed because too few matched: " + "; ".join(res["relaxed"]) + "."
    return Response(txt, [{"type": "recipes", "items": res["items"], "title": "Search results"}])


def _more_results(state, index):
    f = state.filters
    c = prefs_mod.Criteria.from_dict(f.get("criteria")) if f.get("criteria") else None
    n = f.get("top_n") or state.preferences.get("top_n", 5)
    return _run_search(state, index, f.get("dish_tokens", []), f.get("ingredient_terms", []), [], n, f.get("quick", False),
                       f.get("max_steps"), criteria=c, offset=(f.get("offset") or 0) + n)


# ------------------------------------------------------------------ details
def _resolve_ref(text, low, state, index):
    m = ID_RE.search(low)
    if m:
        return (m.group(0), None) if m.group(0) in index.id_to_idx else (None, f"Recipe ID {m.group(0)} was not found.")
    m = REF_NUM.search(text)
    n = int(next(g for g in m.groups() if g)) if m else None
    if n is None:
        mo = REF_ORD.search(low)
        n = ORD[mo.group(1).lower()] if mo else None
    if n is not None:
        if 1 <= n <= len(state.last_result_ids):
            return state.last_result_ids[n - 1], None
        return None, f"There is no result number {n} in the last result list ({len(state.last_result_ids)} shown)."
    if state.active_recipe_id:
        return state.active_recipe_id, None
    if state.last_result_ids:
        return state.last_result_ids[0], "No specific recipe was named, so I used the first result of the last list."
    return None, None


def _do_details(text, low, state, index):
    rid, note = _resolve_ref(text, low, state, index)
    if rid is None:
        return Response(note or "Which recipe? Give a recipe ID (rb_…), say “result 2”, or search first.") if note or not state.last_result_ids else None
    full = index.full(rid)
    state.active_recipe_id = rid
    return Response((note + " " if note else "") + f"Original recipe **{full['title']}** (`{rid}`) — text exactly as stored from `{full['source_file']}`.",
                    [{"type": "details", "recipe": full}])


# ------------------------------------------------------------------ stats
SCOPE_ING_RE = re.compile(r"\b(?:containing|contain|contains|with|that\s+uses?|using|including|includes?|have|has)\s+(?:both\s+|all\s+of\s+|either\s+)?(?P<x>[^?.;!]+)", re.I)
SCOPE_CUT = re.compile(r"(?:,|\b(?:which|what|how|and\s+what|that\s+have|that\s+has|have\s+the|has\s+the|with\s+the)\b).*$", re.I)
TITLE_SCOPE = re.compile(rf"\b(?:among|in|for|of|which|top|first|the)\s+(?:(?:{_NUMRE})\s+)?(?:the\s+|all\s+)?(?P<t>(?!all\b|the\b|these\b|those\b|my\b|your\b|our\b|other\b|many\b|such\b|recipes\b)[a-z]+(?:\s+[a-z]+)?)\s+recipes\b", re.I)


def _stat_scope(index, text, state, count_mode):
    clean = strip_exclusions(text)
    ex = extract_exclusions(text)
    use_active = bool(state.exclusions) and re.search(r"my restrictions|these|those|active|current (?:search|filters)|that fit|compatible", text, re.I)
    eff = list(dict.fromkeys(ex + (state.exclusions if use_active else [])))
    ing_terms = []
    for m in SCOPE_ING_RE.finditer(clean):
        phrase = SCOPE_CUT.sub("", m.group("x"))
        ing_terms += [normalize_ingredient(t) for t in _clean_items(phrase)]
    ing_terms = [t for t in dict.fromkeys(ing_terms) if t]
    title_tokens = []
    mt = TITLE_SCOPE.search(clean)
    if mt:
        title_tokens = [t for t in dish_tokens(mt.group("t")) if t not in {"contain", "containing", "with"}]
    filters, scope = [], None
    if title_tokens:
        tk = set(title_tokens)
        scope = {i for i in range(len(index)) if tk <= index.title_tokens[i]}
        filters.append("title contains: " + ", ".join(title_tokens))
    if not count_mode:
        for t in ing_terms:
            s = index.contains(t)
            scope = s if scope is None else scope & s
            filters.append(f"contains ingredient: {t}")
    if eff:
        ms = [get_matcher(e) for e in eff]
        pool = scope if scope is not None else range(len(index))
        scope = {i for i in pool if index.violation(i, ms) is None}
        filters.append("excluding (text-based): " + ", ".join(describe_matcher(m) for m in ms))
    if not use_active and state.exclusions and not ex:
        filters.append(f"(session restrictions [{', '.join(state.exclusions)}] NOT applied; say “among my restrictions …” to include them)")
    return scope, filters, ing_terms, eff


def _do_stats(text, low, state, index):
    n = parse_top_n(text) or 5
    # percentage follow-up
    if re.search(r"\b(percent|percentage|proportion)\b", low) and state.last_stat and not re.search(r"\b(contain|containing|with|include)\b", low):
        s = state.last_stat
        return Response(f"{s['description']}: {s['count']} of {s['total']} recipes in the whole dataset = **{round(100*s['count']/s['total'], 2)}%** "
                        f"(count ÷ whole-dataset size × 100).")
    if re.search(r"\b(fewest|least|lowest|smallest|shortest|simplest|most|highest|largest|longest)\b.*\bsteps?\b|\b(rank|ranking)\b.*\bsteps?\b", low):
        asc = bool(re.search(r"\b(fewest|least|lowest|smallest|shortest|simplest)\b", low))
        scope, filters, _, _ = _stat_scope(index, text, state, False)
        res = analytics.rank_by_steps(index, scope, n, asc, filters)
        state.intent, state.last_result_ids = "stats", [r["id"] for r in res["rows"]]
        if not res["rows"]:
            return Response("No recipes with usable instructions match that scope, so no ranking is possible.", [{"type": "stats", "result": res}])
        return Response(f"Ranked {res['denominator']} recipes (of {res['n_analyzed']} in scope) by step count, {'fewest' if asc else 'most'} first; top {len(res['rows'])} below.",
                        [{"type": "stats", "result": res}])
    if re.search(r"\b(average|mean|median|typical)\b.*\bingredients?\b|\bingredients?\b.*\b(on average|median)\b", low):
        scope, filters, _, _ = _stat_scope(index, text, state, False)
        res = analytics.ingredient_count_stats(index, scope, filters)
        state.intent = "stats"
        if not res["n_analyzed"]:
            return Response("No recipes match that scope.", [{"type": "stats", "result": res}])
        return Response(f"Over {res['n_analyzed']} recipes: mean **{res['mean']}** ingredient lines, median **{res['median']}** (min {res['min']}, max {res['max']}).", [{"type": "stats", "result": res}])
    if re.search(r"\b(most common|commonly|popular|frequent\w*|top)\b.*\bingredients?\b", low):
        scope, filters, _, _ = _stat_scope(index, text, state, False)
        res = analytics.top_ingredients(index, scope, parse_top_n(text) or 10, filters)
        state.intent = "stats"
        return Response(f"Most common ingredients across {res['n_analyzed']} recipes (share of recipes containing each).", [{"type": "stats", "result": res}])
    if re.search(r"\b(how many|number of|count|percentage|percent|proportion)\b", low):
        scope, filters, terms, _ = _stat_scope(index, text, state, True)
        if not terms:
            return None
        res = analytics.count_containing(index, terms, scope, filters)
        state.intent = "stats"
        state.last_stat = {"description": "Recipes containing " + " and ".join(terms), "count": res["count"], "total": res["whole_dataset_size"]}
        txt = (f"**{res['count']}** of **{res['denominator']}** recipes analysed contain {' and '.join(terms)} = **{res['percentage']}%**.")
        if res["denominator"] != res["whole_dataset_size"]:
            txt += f" ({res['percentage_of_whole_dataset']}% of all {res['whole_dataset_size']} recipes.)"
        return Response(txt, [{"type": "stats", "result": res}])
    return None


# ------------------------------------------------------------------ match
def _do_match(text, low, state, index, new_list):
    p = state.preferences
    note = []
    am = AVAIL_RE.search(text)
    if am and new_list:
        items = parse_available(am.group("x"))
        if items:
            state.available = items
    else:
        rm = REMOVE_AV_RE.search(text)
        if rm:
            names = [normalize_ingredient(t) for t in _clean_items(next(g for g in rm.groups() if g))]
            state.available = [a for a in state.available if a["name"] not in names]
            note.append("removed from your list: " + ", ".join(names))
        ad = ADD_RE.search(text)
        if ad:
            for it in parse_available(ad.group("x")):
                if it["name"] not in [a["name"] for a in state.available]:
                    state.available.append(it)
                    note.append(f"added to your list: {it['name']}")
    added = _add_exclusions(state, extract_exclusions(text))
    for e in added:
        m = get_matcher(e)
        dropped = [a["name"] for a in state.available if m.search(a["name"])]
        if dropped:
            state.available = [a for a in state.available if a["name"] not in dropped]
            note.append(f"since '{e}' is now a hard exclusion, I dropped {', '.join(dropped)} from your available ingredients")
    ms = parse_max_steps(text)
    if ms is not None:
        p["max_steps"] = ms
    tn = parse_top_n(text)
    if tn:
        p["top_n"] = tn
    p.update(parse_preferences(low))
    return _run_match(state, index, note)


def _run_match(state, index, note):
    p = state.preferences
    if not state.available:
        return Response("Which ingredients do you have? e.g. *I have eggs, onions and rice.*", confident=False)
    state.intent = "match"
    res = ingredient_matcher.match_recipes(index, state.available, state.exclusions, p["max_steps"], p["goal"], p["prefer_fewer_steps"],
                                           p["top_n"], p["assume_staples"])
    state.last_result_ids = [r["id"] for r in res["eligible"]]
    state.active_recipe_id = None
    head = (f"Available: {', '.join(res['available'])}. Hard exclusions: {_restr_text(state)}. Step limit: {p['max_steps'] if p['max_steps'] is not None else 'none'}. "
            f"Preference: {res['goal_label']}{' + fewer steps' if p['prefer_fewer_steps'] else ''}. " + ("Updates — " + "; ".join(note) + ". " if note else ""))
    if res["eligible"]:
        txt = head + f"**{res['n_eligible']}** eligible recipe(s) out of {res['n_candidates']} that share at least one of your ingredients; top {len(res['eligible'])} ranked below."
    else:
        txt = head + "**No eligible recipe found** that satisfies every hard constraint. " + ("Closest alternatives are listed separately and are **INELIGIBLE** (they violate a constraint)." if res["alternatives"] else "No near alternatives share your ingredients.")
    return Response(txt, [{"type": "matches", "result": res}])


def _scope_from(index, toks, ing_terms, exclusions):
    scope, filters = None, []
    if toks:
        tk = set(toks)
        scope = {i for i in range(len(index)) if tk <= index.title_tokens[i]}
        filters.append("title contains: " + ", ".join(toks))
    for t in ing_terms:
        s_ = index.contains(t)
        scope = s_ if scope is None else scope & s_
        filters.append(f"contains ingredient: {t}")
    if exclusions:
        ms = [get_matcher(e) for e in exclusions]
        pool = scope if scope is not None else range(len(index))
        scope = {i for i in pool if index.violation(i, ms) is None}
        filters.append("excluding (text-based): " + ", ".join(describe_matcher(m) for m in ms))
    return scope, filters


def _execute_command(cmd, state, index, llm):
    """Execute a schema-validated command interpreted by the LLM.

    The LLM interprets natural language; recipe retrieval, ingredient
    matching, restrictions and analytics remain deterministic and local.
    """
    if not isinstance(cmd, dict):
        return None

    p = state.preferences

    # Update hard exclusions and remove any exclusions the user cleared.
    exclude_items = list(cmd.get("exclude") or [])
    for item in cmd.get("preferences") or []:
        if isinstance(item, str) and item.strip().lower() in ("vegetarian", "vegan"):
            exclude_items.append(item.strip().lower())
    remove_items = cmd.get("remove_exclusions") or []

    added = _add_exclusions(
        state,
        [
            canonical
            for canonical in (
                canonical_exclusion(item)
                for item in exclude_items
                if isinstance(item, str)
            )
            if canonical
        ],
    )
    removed = [
        canonical_exclusion(item)
        for item in remove_items
        if isinstance(item, str)
    ]
    state.exclusions = [item for item in state.exclusions if item not in removed]

    # Persist conversational preferences when supplied.
    max_steps = cmd.get("max_steps")
    if max_steps is not None:
        try:
            p["max_steps"] = max(1, min(100, int(max_steps)))
        except (TypeError, ValueError):
            pass

    top_n = cmd.get("top_n")
    if top_n:
        try:
            p["top_n"] = max(1, min(20, int(top_n)))
        except (TypeError, ValueError):
            pass

    preference = cmd.get("preference")
    if preference == "fewer_steps":
        p["prefer_fewer_steps"] = True
        p["goal"] = "quick"
    elif preference:
        p["goal"] = preference

    cuisine = cmd.get("cuisine")
    meal_type = cmd.get("meal_type")
    food_preferences = cmd.get("preferences") or []

    if cuisine:
        p["cuisine"] = str(cuisine).strip().lower()
    if meal_type:
        p["meal_type"] = str(meal_type).strip().lower()
    if isinstance(food_preferences, list):
        existing = p.get("food_preferences", [])
        if not isinstance(existing, list):
            existing = []
        p["food_preferences"] = list(
            dict.fromkeys(
                existing
                + [str(item).strip().lower() for item in food_preferences if item]
            )
        )

    intent = cmd.get("intent")
    dish = cmd.get("dish") or ""
    toks = dish_tokens(dish)

    include_items = cmd.get("include") or []
    ing = [
        normalized
        for normalized in (
            normalize_ingredient(item)
            for item in include_items
            if isinstance(item, str)
        )
        if normalized
    ]
    quick = bool(cmd.get("quick", False))

    if intent == "restrictions":
        return Response(f"Updated restrictions. Active: {_restr_text(state)}.")

    if intent == "search":
        prev_c = _prev_criteria(state)
        prev_filters = dict(state.filters) if state.intent == "search" else {}
        crit = prefs_mod.Criteria.from_dict({
            "tags": list(cmd.get("tags") or []) + list(cmd.get("preferences") or []),
            "cuisine": cuisine, "meal_type": meal_type,
            "evidence_terms": cmd.get("evidence_terms") or [], "title_keywords": cmd.get("title_keywords") or [],
            "avoid_terms": cmd.get("avoid_terms") or [], "quick": quick})
        crit = prefs_mod.verify_hints(index, crit)
        offset = 0
        n = p.get("top_n", 5)
        if cmd.get("followup") and prev_filters:
            # Conversational follow-up: keep the earlier search and apply only what changed.
            if not toks and not ing:
                toks, ing = prev_filters.get("dish_tokens", []), prev_filters.get("ingredient_terms", [])
            crit = prefs_mod.merge_criteria(prev_c, crit, cmd.get("drop_preferences") or [])
            if max_steps is None:
                max_steps = prev_filters.get("max_steps")
            if cmd.get("more"):
                offset = (prev_filters.get("offset") or 0) + (prev_filters.get("top_n") or n)
                n = prev_filters.get("top_n") or n
        if not toks and not ing and not crit.has_signals():
            return None

        # Cuisine and meal type are NOT verified database columns: they are applied as transparent
        # title/ingredient heuristics (see core/preferences.py) and reported as such in the reply.
        return _run_search(state, index, toks, ing, added, n, quick, max_steps, criteria=crit, offset=offset)

    if intent == "match":
        available_items = cmd.get("available") or []
        parsed_items = [
            item
            for item in (
                ingredient_matcher.parse_item(value)
                for value in available_items
                if isinstance(value, str)
            )
            if item
        ]
        if parsed_items:
            state.available = parsed_items

        for exclusion in state.exclusions:
            matcher = get_matcher(exclusion)
            state.available = [
                item
                for item in state.available
                if not matcher.search(item["name"])
            ]

        note = ["Interpreted by AI"] if parsed_items else []
        return _run_match(state, index, note)

    if intent == "stats":
        stat = cmd.get("stat") or {}
        stat_type = stat.get("type")
        stat_ingredients = stat.get("ingredients") or []
        terms = [
            normalized
            for normalized in (
                normalize_ingredient(item)
                for item in stat_ingredients
                if isinstance(item, str)
            )
            if normalized
        ]

        if stat_type == "count" and terms:
            scope, filters = _scope_from(
                index,
                toks,
                [],
                state.exclusions if added else [],
            )
            result = analytics.count_containing(index, terms, scope, filters or None)
            state.last_stat = {
                "description": "Recipes containing " + " and ".join(terms),
                "count": result["count"],
                "total": result["whole_dataset_size"],
            }
            return Response(
                f"**{result['count']}** of **{result['denominator']}** recipes analysed "
                f"contain {' and '.join(terms)} = **{result['percentage']}%**.",
                [{"type": "stats", "result": result}],
            )

        scope, filters = _scope_from(index, toks, ing + terms, [])
        if stat_type == "top_ingredients":
            result = analytics.top_ingredients(
                index, scope, top_n or 10, filters or None
            )
        elif stat_type == "avg_ingredients":
            result = analytics.ingredient_count_stats(index, scope, filters or None)
        elif stat_type == "rank_steps":
            result = analytics.rank_by_steps(
                index,
                scope,
                top_n or 5,
                stat.get("order") == "fewest",
                filters or None,
            )
        else:
            return None

        state.last_stat = {"type": stat_type, "result": result}
        return Response(
            f"Computed over {result['n_analyzed']} recipes in scope.",
            [{"type": "stats", "result": result}],
        )

    if intent == "substitute":
        substitute_data = cmd.get("substitute") or {}
        ingredient = substitute_data.get("ingredient")
        recipe = substitute_data.get("recipe")
        if not ingredient:
            return None
        text = f"Replace {ingredient}"
        if recipe:
            text += f" in the {recipe} recipe"
        return _do_substitution(text, text.lower(), state, index, llm)

    return None


# ------------------------------------------------------------------ substitution
def _parse_target(text):
    for rx, forced in SUB_PATTERNS:
        m = rx.search(text)
        if m:
            t = re.sub(r"^(?:the|all|any|my)\s+", "", m.group("t").strip().lower())
            if t and t not in GENERIC_TITLE and len(t.split()) <= 3:
                verb = forced or (m.groupdict().get("verb") or "").lower()
                kind = "replace" if forced == "replace" or re.match(r"replace|substitute|swap|instead|in place|replacement|alternative", verb or "") else "remove"
                return t, kind
    return None, None


def _resolve_sub_recipe(text, low, state, index, target):
    m = ID_RE.search(low)
    if m:
        return (m.group(0), None) if m.group(0) in index.id_to_idx else (None, f"Recipe ID {m.group(0)} was not found.")
    if REF_NUM.search(text) or REF_ORD.search(low):
        rid, msg = _resolve_ref(text, low, state, index)
        if rid:
            return rid, msg
    mt = TITLE_RE.search(text)
    title = mt.group("t").strip().lower() if mt else ""
    if title and title not in GENERIC_TITLE:
        toks = set(dish_tokens(title))
        act = state.active_recipe_id
        if act and toks and toks <= index.title_tokens[index.id_to_idx[act]]:
            return act, None
        res = search_recipes(index, list(toks), (), (), limit=500)
        if not res["total"]:
            return None, f"I couldn't find a recipe titled like “{title}”. Try a different name or give a recipe ID."
        with_t = index.contains(target) if target else None
        ids = [r["id"] for r in res["items"]]
        pick = next((r for r in ids if with_t is None or index.id_to_idx[r] in with_t), None)
        if pick is None:
            pick = ids[0]
            msg = f"None of the {res['total']} recipes matching “{title}” (first 500 checked) clearly lists '{target}'; using the top title match."
        else:
            msg = f"Using the best title match that contains '{target}' (`{pick}`); {res['total']} recipe(s) matched “{title}”. Give a recipe ID to choose another."
        return pick, msg
    if state.active_recipe_id:
        return state.active_recipe_id, None
    if state.last_result_ids:
        return state.last_result_ids[0], "No recipe was named, so I used the first result of the last list."
    return None, "__NONE__"


def _do_substitution(text, low, state, index, llm=None):
    target, kind = _parse_target(text)
    if not target:
        return Response("Which ingredient should be replaced or removed? e.g. *Remove eggs from the banana bread recipe.*", confident=False)
    _add_exclusions(state, extract_exclusions(text))
    if kind == "remove" and canonical_exclusion(target) not in state.exclusions:
        _add_exclusions(state, [canonical_exclusion(target)])
    rid, note = _resolve_sub_recipe(text, low, state, index, target)
    if rid is None:
        if note == "__NONE__":
            return _general_sub(target, state, index, llm)
        return Response(note, confident=False)
    restr = list(state.exclusions)
    if kind == "replace":
        restr.append(canonical_exclusion(target))
    state.intent, state.active_recipe_id = "substitution", rid
    state.last_substitution = {"recipe_id": rid, "targets": [target], "kind": kind}
    return _render_sub(index, rid, [target], restr, note, llm)


def _general_sub(target, state, index, llm):
    restr = list(state.exclusions)
    res = substitution.general_substitution(index, target, restr)
    if llm and llm.available:
        extra = sub_reasoning.general_with_llm(llm, index, target, res["options"], restr)
        if extra:
            res = substitution.general_substitution(index, target, restr, extra)
            res["ai_notes"] = {e["name"]: e["note"] for e in extra}
    state.intent = "substitution"
    if not res["options"]:
        return Response(f"I couldn't find dataset-supported alternatives for **{target}** (under your restrictions: {_restr_text(state)}). "
                        "Name a recipe and I will look for evidence in similar recipes.", [{"type": "general_sub", "result": res}], confident=False)
    return Response(f"Alternatives for **{target}** that exist in the dataset ({res['n_recipes_with']:,} recipes use it). Evidence is **{res['evidence_class']}** "
                    "because no specific recipe was named — say e.g. “replace it in the penne pasta recipe” for a full adaptation.", [{"type": "general_sub", "result": res}])


def _redo_substitution(text, state, index, llm=None):
    _add_exclusions(state, extract_exclusions(text))
    ls = state.last_substitution
    return _render_sub(index, ls["recipe_id"], ls["targets"], list(state.exclusions), "Re-ran the same adaptation with your updated restrictions.", llm)


def _render_sub(index, rid, targets, restr, note, llm=None):
    res = substitution.propose_substitution(index, rid, targets, restr)
    if res.get("error"):
        return Response(res["error"], [])
    sub_reasoning.refine(llm, index, res, restr, targets[0])
    ev = res["overall_evidence"]
    if res["revision_supported"]:
        txt = (f"Proposed adaptation of **{res['recipe']['title']}** (`{rid}`). Evidence: **{ev}**"
               f"{' (AI-reasoned over retrieved recipes, validated by code)' if res['llm'].get('used') else ' (deterministic, from similar local recipes)'}. "
               "Original and proposed text are shown separately; effects are predictions, not verified outcomes.")
    else:
        txt = (f"I could **not** find enough support in the dataset for a reliable adaptation of **{res['recipe']['title']}** (`{rid}`) — evidence: **{ev}**. "
               "No revised recipe is offered; unverified ideas are listed separately.")
    return Response((note + " " if note and note != "__NONE__" else "") + txt, [{"type": "substitution", "result": res}])


# ------------------------------------------------------------------ simple recipe facts
def _do_recipe_fact(text, low, state, index):
    pat = re.search(r"how many (?:instruction |cooking )?steps|how many ingredients|\b(?:does|do|is)\b.*\b(?:contain|have|use|include|has)\b\s+(?P<x>[a-z][a-z \-]*)", low)
    if not pat:
        return None
    rid, note = _resolve_ref(text, low, state, index)
    if rid is None:
        return None
    i = index.id_to_idx[rid]
    title = index.titles[i]
    if "how many" in pat.group(0):
        if "ingredients" in pat.group(0):
            return Response(f"**{title}** (`{rid}`) lists **{index.ing_counts[i]}** ingredient lines ({len(index.names[i])} distinct normalised names).")
        n = index.steps[i]
        return Response(f"**{title}** (`{rid}`) has **{n if n else 'an unknown number of'}** instruction steps under the counting rule (non-empty lines, or sentences if one line).")
    term = re.sub(r"\b(it|this|that|recipe|one|any|the)\b", " ", pat.group("x")).strip()
    term = re.split(r"\b(?:and|or|\?)\b|[?.,]", term)[0].strip()
    if not term:
        return None
    m = get_matcher(term)
    full = index.full(rid)
    hits = [l for l in full["ingredients"] if m.search(l.lower())]
    if hits:
        return Response(f"Yes — **{title}** (`{rid}`) lists: " + "; ".join(hits[:5]) + f" (matched “{m.label}” in the ingredient text).")
    return Response(f"No ingredient line of **{title}** (`{rid}`) mentions “{m.label}”. (Text-based check only; hidden ingredients in products can't be seen.)")
