
"""Title / ingredient search (deterministic, in-memory over the SQLite-derived index)."""
from __future__ import annotations

import re

from .ingredient_normalizer import get_matcher, singularize
from . import preferences as prefs_mod


# Words that should not be treated as recipe or ingredient search terms.
STOPWORDS = {
    "find", "a", "an", "the", "recipe", "recipes", "for", "me", "with",
    "some", "show", "give", "please", "i", "im", "m", "am", "want",
    "need", "looking", "search", "dish", "dishes", "make", "how", "to",
    "using", "that", "contains", "containing", "and", "any", "good",
    "best", "easy", "can", "you", "get", "of", "in", "on", "my", "is",
    "are", "something", "like", "also", "now", "just", "suggest", "tell",
    "ready", "high", "low", "dinner", "lunch", "breakfast", "meal",
    "quick", "fast", "simple", "speedy", "weeknight", "craving",
    "tonight", "idea", "ideas", "about", "have", "has", "do", "does",
    "would", "could", "let", "us", "it", "this", "one", "want",
    "prefer", "preferably", "hungry", "mood", "today", "please",
    "something", "delicious", "tasty", "food", "eat", "eating"
}

_WORD = re.compile(r"[a-z]+")


# Common spelling variations and transliterations.
DISH_ALIASES = {
    "pharata": "paratha",
    "parata": "paratha",
    "parotha": "paratha",
    "parantha": "paratha",
    "pratha": "paratha",
}


def dish_tokens(text: str) -> list[str]:
    """
    Convert a natural-language query into meaningful search tokens.

    Examples:
        'I'm craving pasta' -> ['pasta']
        'Find a quick pasta recipe' -> ['pasta']
        'aloo pharata' -> ['aloo', 'paratha']
    """
    out = []

    for word in _WORD.findall(text.lower()):
        if word in STOPWORDS:
            continue

        token = singularize(word)

        # Correct common spelling variations.
        token = DISH_ALIASES.get(token, token)

        if token and token not in out:
            out.append(token)

    return out


def search_recipes(
    index,
    tokens,
    ingredient_terms=(),
    exclusions=(),
    limit=5,
    max_steps=None,
    prefer_short=False,
    criteria=None,
    offset=0
) -> dict:

    tokens = list(tokens)

    matchers = [get_matcher(e) for e in exclusions]

    result = {
        "items": [],
        "total": 0,
        "total_before_exclusions": 0,
        "excluded_count": 0,
        "scope": "title",
        "dish_tokens": tokens,
        "ingredient_terms": list(ingredient_terms),
        "exclusions": [m.label for m in matchers],
        "relaxed": [],
        "total_exact": True,
        "criteria": criteria.to_dict() if criteria is not None else None
    }

    prefs = criteria if (criteria is not None and criteria.has_signals()) else None

    # Apply ingredient filters.
    pool = None

    for term in ingredient_terms:
        matches = index.contains(term)
        pool = matches if pool is None else (pool & matches)

    if not tokens and pool is None and prefs is None:
        return result

    if prefs is not None and not tokens and pool is None:
        return _preference_only(index, result, prefs, matchers, limit, offset, max_steps)

    scan = sorted(pool) if pool is not None else range(len(index))

    # Search recipe titles first.
    if tokens:
        token_set = set(tokens)

        candidates = [
            i for i in scan
            if token_set <= index.title_tokens[i]
        ]

        # If no exact title match exists, search title and ingredients.
        if not candidates:
            result["scope"] = "title_or_ingredients"
            candidates = []

            for i in scan:
                missing = token_set - index.title_tokens[i]

                if not missing or all(
                    any(
                        term in ingredient.split()
                        for ingredient in index.names[i]
                    )
                    for term in missing
                ):
                    candidates.append(i)

    else:
        candidates = list(scan)
        result["scope"] = "ingredients"

    # Descriptive preferences (spicy, light, cuisine, meal type ...) re-rank and filter the base matches.
    if prefs is not None:
        result["scope"] = "preferences+" + result["scope"]
        base = set(candidates)
        res = prefs_mod.retrieve(index, prefs, matchers, max_steps, base=base, need=limit + offset)
        result["total_before_exclusions"] = res["pool_size"]
        result["total"] = len(res["order"]) if res["exact"] else res["pool_size"]
        result["excluded_count"] = res["excluded"] or 0
        result["total_exact"] = res["exact"]
        result["relaxed"] = res["relaxed"]
        return _emit(index, result, res["order"][offset:offset + limit], matchers, prefs, res["scores"])

    # Apply maximum cooking-step restriction.
    if max_steps is not None:
        candidates = [
            i for i in candidates
            if 0 < index.steps[i] <= max_steps
        ]

    result["total_before_exclusions"] = len(candidates)

    # Apply exclusion restrictions.
    if matchers:
        candidates = [
            i for i in candidates
            if index.violation(i, matchers) is None
        ]

    result["total"] = len(candidates)

    result["excluded_count"] = (
        result["total_before_exclusions"] - len(candidates)
    )

    # Rank matching recipes.
    phrase = " ".join(tokens)

    if prefer_short:
        candidates.sort(
            key=lambda i: (
                index.steps[i] if index.steps[i] > 0 else 10**6,
                len(index.titles[i]),
                index.titles[i].lower(),
                index.ids[i]
            )
        )
    else:
        candidates.sort(
            key=lambda i: (
                0 if phrase and phrase in index.titles[i].lower() else 1,
                len(index.title_tokens[i]),
                len(index.titles[i]),
                index.titles[i].lower(),
                index.ids[i]
            )
        )

    # Return matching recipes.
    return _emit(index, result, candidates[offset:offset + limit], matchers, None, None)


def _preference_only(index, result, prefs, matchers, limit, offset, max_steps):
    """No dish name or ingredient: retrieve purely from descriptive preferences (Gemini/lexicon criteria)."""
    result["scope"] = "preferences"
    res = prefs_mod.retrieve(index, prefs, matchers, max_steps, base=None, need=limit + offset)
    result["total_before_exclusions"] = res["pool_size"]
    result["total"] = len(res["order"]) if res["exact"] else res["pool_size"]
    result["excluded_count"] = res["excluded"] or 0
    result["total_exact"] = res["exact"]
    result["relaxed"] = res["relaxed"]
    return _emit(index, result, res["order"][offset:offset + limit], matchers, prefs, res["scores"])


def _emit(index, result, idxs, matchers, prefs, scores):
    for i in idxs:
        item = {
            "id": index.ids[i],
            "title": index.titles[i],
            "steps": index.steps[i],
            "ingredient_lines": index.ing_counts[i],
            "restriction_check": (
                "checked ingredient list for: "
                + ", ".join(m.label for m in matchers)
                + " — none found (text-based check)"
            ) if matchers else None
        }
        if prefs is not None:
            item["why"] = prefs_mod.explain(index, i, prefs)
            if scores is not None and i in scores:
                item["pref_score"] = round(scores[i], 2)
        result["items"].append(item)
    return result
