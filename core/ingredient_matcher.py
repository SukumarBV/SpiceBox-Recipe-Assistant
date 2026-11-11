"""Ingredient-based recipe matching with hard constraints and a fully reproducible score.

AUDIT NOTE (v1 bug): v1 scored 100*(w_usage*usage + w_coverage*coverage). With the 'fewest missing' weights
(0.3/0.7) a recipe using 1 of 4 ingredients with 100% coverage scored 100*(0.3*0.25+0.7*1.0)=77.5 and beat
recipes that used 3-4 of the user's ingredients. The displayed numbers were consistent but the formula ignored
the explicit goal 'use as many as possible'. v2 uses four separate 0-100 components, preference-dependent weights,
and a relevance gate (see FORMULA). The displayed components + weights always reproduce the displayed score.
"""
from __future__ import annotations

from .ingredient_normalizer import get_matcher, is_variant, normalize_ingredient, split_quantity

STAPLES = {"salt", "water", "pepper", "black pepper", "ground black pepper", "kosher salt", "sea salt"}
VARIANT_CREDIT = 0.7
MISSING_CAP = 10          # >= 10 additional ingredients -> missing component 0
STEP_CAP = 20
# weights: (usage, coverage, missing, steps)
PREF_WEIGHTS = {
    "balanced": (0.50, 0.20, 0.25, 0.05),
    "most_used": (0.65, 0.15, 0.15, 0.05),
    "fewest_missing": (0.30, 0.20, 0.45, 0.05),
    "most_used+fewest_missing": (0.45, 0.15, 0.35, 0.05),
    "closest": (0.35, 0.40, 0.20, 0.05),
}
STEP_WEIGHT_FEWER = 0.25
GOAL_LABELS = {"balanced": "Balanced", "most_used": "Use as many available ingredients as possible",
               "fewest_missing": "Fewest additional ingredients", "most_used+fewest_missing": "Use as many as possible, then fewest additional",
               "closest": "Closest recipe match"}

FORMULA = (
    "Four components, each 0-100: USAGE = 100 × (your n ingredients used) ÷ n; COVERAGE = 100 × (match credit over the recipe's "
    "distinct ingredient names) ÷ (number of names), credit 1.0 exact / 0.7 variant such as 'red onion' for 'onion'; "
    f"MISSING = 100 × (1 − additional_ingredients ÷ {MISSING_CAP}), floored at 0; STEPS = 100 × (1 − (min(steps,{STEP_CAP}) − 1) ÷ {STEP_CAP - 1}) "
    "(0 if the step count is unknown). Score = w_usage×USAGE + w_coverage×COVERAGE + w_missing×MISSING + w_steps×STEPS "
    "with preference-dependent weights (shown on every card). 'Fewer steps' moves 25% of the weight to STEPS. "
    "Relevance gate: a recipe that uses fewer than min(2, n) of your ingredients is ranked after every recipe that does, "
    "so a one-ingredient recipe cannot outrank richer matches. Hard constraints (exclusions, step limit) are applied BEFORE ranking. "
    "Ties: fewer additional ingredients → fewer steps → title → ID."
)


def weights_for(goal: str, prefer_fewer_steps: bool) -> tuple:
    w = list(PREF_WEIGHTS.get(goal, PREF_WEIGHTS["balanced"]))
    if prefer_fewer_steps:
        scale = (1 - STEP_WEIGHT_FEWER) / (1 - w[3])
        w = [x * scale for x in w[:3]] + [STEP_WEIGHT_FEWER]
    return tuple(round(x, 4) for x in w)


def combine(components: dict, weights: tuple) -> float:
    """Exactly how the score is computed from the DISPLAYED components (so users can re-derive it)."""
    return round(weights[0] * components["usage"] + weights[1] * components["coverage"]
                 + weights[2] * components["missing"] + weights[3] * components["steps"], 2)


def parse_item(text: str) -> dict | None:
    name = normalize_ingredient(text)
    if not name:
        return None
    qty, unit, _ = split_quantity(text)
    return {"name": name, "qty": (qty + " " + unit).strip()}


def _variant_names(index, a: str):
    toks = a.split()
    first = min(toks, key=lambda t: len(index.token_index.get(t, ())))
    for n in index.token_names.get(first, ()):
        if n != a and is_variant(a, n):
            yield n


def match_recipes(index, available, exclusions=(), max_steps=None, goal="balanced", prefer_fewer_steps=False,
                  top_n=5, assume_staples=True) -> dict:
    user = []
    for it in available:
        n = normalize_ingredient(it["name"]) if isinstance(it, dict) else normalize_ingredient(it)
        if n and n not in [u["name"] for u in user]:
            user.append({"name": n, "qty": it.get("qty", "") if isinstance(it, dict) else ""})
    weights = weights_for(goal, prefer_fewer_steps)
    out = {"available": [u["name"] for u in user], "quantities_given": [f"{u['qty']} {u['name']}" for u in user if u["qty"]],
           "exclusions": [], "max_steps": max_steps, "goal": goal, "goal_label": GOAL_LABELS.get(goal, goal),
           "prefer_fewer_steps": prefer_fewer_steps, "weights": dict(zip(("usage", "coverage", "missing", "steps"), weights)),
           "formula": FORMULA, "eligible": [], "alternatives": [], "n_candidates": 0, "n_eligible": 0, "n_partial_only": 0,
           "excluded_by_steps": 0, "excluded_by_exclusion": 0, "assume_staples": assume_staples,
           "staples": sorted(STAPLES) if assume_staples else [],
           "notes": ["Name-level match only: quantity sufficiency was NOT verified (recipe quantities are free text and not reliably comparable)."]}
    if out["quantities_given"]:
        out["notes"].append("You gave quantities (" + ", ".join(out["quantities_given"]) + "); they were not used for ranking because recipe amounts could not be compared reliably.")
    matchers = [get_matcher(e) for e in exclusions]
    out["exclusions"] = [m.label for m in matchers]
    if not user:
        out["notes"].append("No available ingredients were given.")
        return out
    all_avail = [(u["name"], False) for u in user]
    if assume_staples:
        all_avail += [(s, True) for s in sorted(STAPLES) if s not in [u["name"] for u in user]]
    match_map: dict[str, tuple] = {}
    for ai, (a, _) in enumerate(all_avail):
        if a in index.name_index:
            match_map.setdefault(a, (ai, 1.0, "exact"))
    for ai, (a, _) in enumerate(all_avail):
        if a and all(t in index.token_index for t in a.split()):
            for n in _variant_names(index, a):
                match_map.setdefault(n, (ai, VARIANT_CREDIT, "variant"))
    cand = set()
    for n, (ai, _, _) in match_map.items():
        if ai < len(user):                      # candidates must share >=1 of the USER's ingredients (not just salt)
            cand.update(index.name_index.get(n, ()))
    out["n_candidates"] = len(cand)
    n_user = len(user)
    min_used = min(2, n_user)

    def score(i):
        names = index.names[i]
        matched, used, credit = [], set(), 0.0
        for n in names:
            mm = match_map.get(n)
            if mm:
                ai, cr, kind = mm
                credit += cr
                used.add(ai)
                matched.append({"recipe_ingredient": n, "your_ingredient": all_avail[ai][0], "match": kind,
                                "pantry_staple_assumed": all_avail[ai][1]})
        R = len(names) or 1
        used_user = {a for a in used if a < n_user}
        matched_names = {m["recipe_ingredient"] for m in matched}
        missing = [n for n in names if n not in matched_names]
        steps = index.steps[i]
        comps = {"usage": round(100 * len(used_user) / n_user, 2), "coverage": round(100 * credit / R, 2),
                 "missing": round(100 * max(0.0, 1 - len(missing) / MISSING_CAP), 2),
                 "steps": round(100 * (1 - (min(steps, STEP_CAP) - 1) / (STEP_CAP - 1)), 2) if steps > 0 else 0.0}
        s = combine(comps, weights)
        unused = [user[a]["name"] for a in range(n_user) if a not in used_user]
        reason = (f"uses {len(used_user)}/{n_user} of your ingredients; covers {credit:.1f}/{len(names)} of the recipe's ingredient names; "
                  f"{len(missing)} additional ingredient(s) needed; {steps if steps else 'unknown'} steps. "
                  f"Score {s} = {weights[0]}×{comps['usage']} + {weights[1]}×{comps['coverage']} + {weights[2]}×{comps['missing']} + {weights[3]}×{comps['steps']}")
        return {"id": index.ids[i], "title": index.titles[i], "score": s, "steps": steps, "n_used": len(used_user),
                "usage": comps["usage"], "coverage": comps["coverage"], "components": comps, "weights": out["weights"],
                "matched": matched, "missing": missing, "unused_available": unused,
                "ingredient_lines": index.ing_counts[i], "ingredient_names": len(names),
                "partial_only": len(used_user) < min_used, "reason": reason}

    def sort_key(r):
        return (1 if r["partial_only"] else 0, -r["score"], len(r["missing"]), r["steps"] if r["steps"] > 0 else 10**6, r["title"].lower(), r["id"])

    elig, inelig = [], []
    for i in sorted(cand):
        reasons = []
        if max_steps is not None:
            s = index.steps[i]
            if s == 0:
                reasons.append("step count unknown (cannot verify the step limit)")
                out["excluded_by_steps"] += 1
            elif s > max_steps:
                reasons.append(f"{s} steps exceeds the limit of {max_steps}")
                out["excluded_by_steps"] += 1
        v = index.violation(i, matchers) if matchers else None
        if v:
            reasons.append(f"contains '{v[1]}' (violates exclusion: {v[0]})")
            out["excluded_by_exclusion"] += 1
        (inelig if reasons else elig).append((i, reasons))
    out["n_eligible"] = len(elig)
    ranked = sorted((score(i) for i, _ in elig), key=sort_key)
    out["n_partial_only"] = sum(r["partial_only"] for r in ranked)
    out["eligible"] = ranked[:top_n]
    if any(r["partial_only"] for r in out["eligible"]):
        out["notes"].append(f"Fewer than {top_n} eligible recipes use at least {min_used} of your ingredients, so some listed recipes (marked 'partial') use only one.")
    if not ranked:
        alts = []
        for i, reasons in inelig:
            r = score(i)
            r["violations"] = reasons
            alts.append(r)
        alts.sort(key=sort_key)
        out["alternatives"] = alts[:3]
        out["notes"].append("NO ELIGIBLE RECIPE: no recipe satisfies every hard constraint. Any alternatives listed separately are INELIGIBLE and violate at least one constraint.")
    return out
