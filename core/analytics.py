"""Deterministic dataset statistics. Every function reports filters, n analysed, denominator,
normalisation rule and calculation method. No LLM is involved."""
from __future__ import annotations
import statistics
from collections import Counter

NORM_RULE = ("Ingredient lines are lower-cased; quantities, units, bracketed notes, text after the first comma and "
             "preparation/size words (chopped, fresh, large, …) are removed; words are singularised (tomatoes→tomato). "
             "Product-changing words are kept, so 'tomato paste' and 'canned tomato' stay distinct names.")
TOKEN_RULE = ("A recipe 'contains X' if any normalised ingredient name contains X as a whole-word phrase "
              "(e.g. 'tomato sauce' counts for 'tomato'; 'tomatillo' does not).")
STEP_RULE = ("Steps = non-empty instruction lines (or list items). If the instructions are one single line, sentences are "
             "counted instead. Recipes with no usable instructions have 0 steps and are excluded from step rankings.")


def _base(index, scope):
    return sorted(scope) if scope is not None else list(range(len(index)))


def _pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def count_containing(index, terms, scope=None, filters=None) -> dict:
    base = _base(index, scope)
    bset = set(base)
    tok = set.intersection(*[index.contains(t, "token") for t in terms]) & bset if terms else set()
    ext = set.intersection(*[index.contains(t, "exact") for t in terms]) & bset if terms else set()
    total = len(index)
    return {
        "kind": "count", "terms": list(terms), "filters": filters or ["none (whole dataset)"],
        "n_analyzed": len(base), "denominator": len(base), "count": len(tok),
        "percentage": _pct(len(tok), len(base)), "percentage_of_whole_dataset": _pct(len(tok), total),
        "whole_dataset_size": total,
        "exact_name_count": len(ext), "exact_name_percentage": _pct(len(ext), len(base)),
        "sample_ids": [index.ids[i] for i in sorted(tok, key=lambda i: index.ids[i])[:5]],
        "normalization": NORM_RULE + " " + TOKEN_RULE,
        "method": f"count(recipes in scope where ALL of {list(terms)} are contained) ÷ recipes in scope × 100. "
                  "Recipe-level presence (a recipe counts once). Exact-name count shown as a stricter comparison.",
        "notes": ["Exact duplicate records (if any) are counted as separate recipes; see ingestion_report.json."],
    }


def top_ingredients(index, scope=None, n=10, filters=None) -> dict:
    base = _base(index, scope)
    c = Counter()
    for i in base:
        c.update(index.names[i])
    rows = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return {
        "kind": "top_ingredients", "filters": filters or ["none (whole dataset)"], "n_analyzed": len(base), "denominator": len(base),
        "rows": [{"rank": r, "ingredient": k, "recipes": v, "percent": _pct(v, len(base))} for r, (k, v) in enumerate(rows, 1)],
        "normalization": NORM_RULE,
        "method": "For each normalised ingredient name, count recipes containing it (once per recipe); sort by count desc, then name.",
        "notes": ["Staples such as salt or water usually dominate; this is a property of the data, not a ranking of 'importance'."],
    }


def ingredient_count_stats(index, scope=None, filters=None) -> dict:
    base = _base(index, scope)
    vals = [index.ing_counts[i] for i in base]
    if not vals:
        return {"kind": "ingredient_stats", "filters": filters or [], "n_analyzed": 0, "denominator": 0, "mean": None, "median": None,
                "min": None, "max": None, "normalization": "", "method": "", "notes": ["No recipes in scope."]}
    return {
        "kind": "ingredient_stats", "filters": filters or ["none (whole dataset)"], "n_analyzed": len(vals), "denominator": len(vals),
        "mean": round(statistics.fmean(vals), 3), "median": statistics.median(vals), "min": min(vals), "max": max(vals),
        "normalization": "Ingredient count = number of non-empty ingredient lines in the source list (boilerplate lines such as 'ADVERTISEMENT' ignored). Lines are not de-duplicated.",
        "method": "Arithmetic mean and median of the ingredient-line count over all recipes in scope.",
        "notes": [],
    }


def rank_by_steps(index, scope=None, n=5, ascending=True, filters=None) -> dict:
    base = _base(index, scope)
    known = [i for i in base if index.steps[i] > 0]
    key = (lambda i: (index.steps[i], index.titles[i].lower(), index.ids[i])) if ascending else \
          (lambda i: (-index.steps[i], index.titles[i].lower(), index.ids[i]))
    known.sort(key=key)
    top = known[:n]
    notes = [f"{len(base) - len(known)} recipes in scope have no usable instructions (0 steps) and were excluded from the ranking."]
    if top and len(known) > n:
        cutoff = index.steps[top[-1]]
        tied = sum(1 for i in known if index.steps[i] == cutoff)
        if tied > sum(1 for i in top if index.steps[i] == cutoff):
            notes.append(f"{tied} recipes share the cut-off value of {cutoff} steps; ties are broken by title (A→Z) then recipe ID, so the shown set is one valid choice among equal ones.")
    return {
        "kind": "rank_steps", "filters": filters or ["none (whole dataset)"], "n_analyzed": len(base), "denominator": len(known),
        "ascending": ascending,
        "rows": [{"rank": r, "id": index.ids[i], "title": index.titles[i], "steps": index.steps[i]} for r, i in enumerate(top, 1)],
        "normalization": STEP_RULE, "method": f"Sort recipes in scope by step count ({'ascending' if ascending else 'descending'}), tie-break title then ID; take top {n}.",
        "notes": notes,
    }
