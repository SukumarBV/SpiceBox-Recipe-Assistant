
"""Title / ingredient search (deterministic, in-memory over the SQLite-derived index)."""
from __future__ import annotations

import re

from .ingredient_normalizer import get_matcher, singularize


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
    prefer_short=False
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
        "exclusions": [m.label for m in matchers]
    }

    # Apply ingredient filters.
    pool = None

    for term in ingredient_terms:
        matches = index.contains(term)
        pool = matches if pool is None else (pool & matches)

    if not tokens and pool is None:
        return result

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
    for i in candidates[:limit]:
        result["items"].append({
            "id": index.ids[i],
            "title": index.titles[i],
            "steps": index.steps[i],
            "ingredient_lines": index.ing_counts[i],
            "restriction_check": (
                "checked ingredient list for: "
                + ", ".join(m.label for m in matchers)
                + " — none found (text-based check)"
            ) if matchers else None
        })

    return result
