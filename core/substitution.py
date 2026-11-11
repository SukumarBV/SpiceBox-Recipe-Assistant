"""Evidence-based ingredient substitution grounded primarily in the local dataset.

Method (all deterministic):
 1. Find the ingredient line(s) to change (target ingredient and/or lines violating a restriction).
 2. Find 'neighbour' recipes: high ingredient overlap (Jaccard >= 0.25, target ingredient ignored).
 3. Split neighbours into those WITH the target ingredient and those WITHOUT it.
 4. Candidate substitutes = ingredients over-represented in neighbours WITHOUT the target (lift vs neighbours with it),
    not already in the recipe and not violating any restriction  -> DIRECT local evidence.
 5. If no direct candidate: a small curated list of commonly used alternatives (general culinary knowledge) is checked
    against the dataset -> INDIRECT evidence (clearly labelled; only ingredients present in the dataset are offered).
 6. Omission is offered only if enough neighbours do without the ingredient.
Effects on taste/texture are PREDICTIONS from general culinary knowledge, never verified outcomes.
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict

from .ingredient_normalizer import get_matcher, normalize_ingredient, phrase_in, related
from .sub_reasoning import evidence_class

DISCLAIMER = ("This is a text-based adaptation from recipe data. It does NOT establish that the result is medically safe or "
              "allergen-free (hidden ingredients, cross-contact and product formulations are not visible in recipe text). "
              "Check labels and consult a qualified professional for allergies.")

ROLES = {
    "egg": ("binding, moisture and structure/lift", "may produce a denser, more crumbly or less risen result"),
    "butter": ("fat, richness, tenderness and browning", "may change richness, browning and crumb/tenderness"),
    "milk": ("liquid and mild richness", "may change richness and browning; thickness may need adjusting"),
    "cream": ("fat, richness and body in sauces", "sauce may be thinner or less rich; reduction time may change"),
    "buttermilk": ("acidity, moisture and tenderness", "may lose tang and some tenderness"),
    "cheese": ("salty/umami flavour, fat and melting/binding", "flavour will be less salty/savoury and texture less creamy"),
    "yogurt": ("moisture, acidity and tenderness", "may be less tangy and slightly drier or wetter"),
    "flour": ("structure and thickening", "structure and thickening power may change noticeably"),
    "sugar": ("sweetness, moisture and browning", "sweetness, browning and texture may change"),
    "oil": ("fat and moisture", "may change richness and flavour"),
    "mayonnaise": ("fat, binding and tang", "may be less creamy or tangy"),
    "honey": ("sweetness and moisture", "sweetness and moisture may change"),
}
HINTS = {
    "egg": ["applesauce", "banana", "yogurt", "ground flaxseed", "silken tofu", "vinegar", "cornstarch"],
    "butter": ["olive oil", "vegetable oil", "margarine", "coconut oil", "shortening"],
    "milk": ["soy milk", "almond milk", "oat milk", "coconut milk", "water"],
    "cream": ["coconut cream", "coconut milk", "soy milk", "cashew"],
    "buttermilk": ["soy milk", "vinegar", "lemon juice"],
    "cheese": ["nutritional yeast", "tofu"],
    "yogurt": ["coconut milk", "silken tofu", "soy yogurt"],
    "mayonnaise": ["olive oil", "avocado", "mustard"],
    "sugar": ["honey", "maple syrup", "brown sugar"],
    "honey": ["maple syrup", "sugar"],
    "flour": ["rice flour", "cornstarch", "oat flour"],
}
STAPLE_NAMES = {"salt", "water", "pepper", "black pepper", "sugar", "flour", "all purpose flour", "oil", "vegetable oil"}


def _role_key(T: str, table: dict):
    toks = T.split()
    for tk in reversed(toks):
        if tk in table:
            return tk
    return None


def find_neighbors(index, idx0: int, ignore: set, min_jaccard=0.25, max_candidates=4000):
    base = set(index.names[idx0]) - ignore
    if not base:
        return []
    sized = sorted((len(index.name_index[n]), n) for n in base if 0 < len(index.name_index.get(n, ())) <= 30000)
    rare = [n for _, n in sized[:4]]
    if not rare:
        return []
    counts = Counter()
    for n in rare:
        counts.update(index.name_index[n])
    counts.pop(idx0, None)
    need = 2 if len(rare) >= 3 else 1
    cands = sorted((j for j, c in counts.items() if c >= need), key=lambda j: (-counts[j], index.ids[j]))[:max_candidates]
    out = []
    for j in cands:
        other = set(index.names[j]) - ignore
        union = len(base | other)
        jac = len(base & other) / union if union else 0.0
        if jac >= min_jaccard:
            out.append((jac, j))
    out.sort(key=lambda x: (-x[0], index.ids[x[1]]))
    return out


def _step_example(index, rid: str, name: str):
    full = index.full(rid)
    if not full:
        return None
    head = name.split()[-1]
    rx = re.compile(rf"\b{re.escape(head)}(?:s|es)?\b", re.I)
    for s in full["instructions_steps"]:
        if rx.search(s):
            return {"recipe_id": rid, "step": s[:240] + ("…" if len(s) > 240 else "")}
    return None


def analyse_slot(index, idx0, T, restr_matchers, orig_names, line_idx, lines):
    ignore = {n for n in orig_names if related(T, n)}
    neigh = find_neighbors(index, idx0, ignore)
    with_t, without_t = [], []
    for jac, j in neigh:
        (with_t if any(related(T, n) for n in index.names[j]) else without_t).append((jac, j))
    cnt, where = Counter(), defaultdict(list)
    for jac, j in without_t:
        for n in index.names[j]:
            if n not in orig_names and not related(T, n):
                cnt[n] += 1
                where[n].append(j)
    cnt_with = Counter(n for _, j in with_t for n in index.names[j])
    cands = []
    for n, c in cnt.items():
        if c < 2 or n in STAPLE_NAMES or len(n.split()) > 5:
            continue
        if any(m.search(n) for m in restr_matchers):
            continue
        share = c / len(without_t)
        share_with = cnt_with.get(n, 0) / len(with_t) if with_t else 0.0
        lift = share / (share_with + 0.05)
        if with_t and lift < 1.5:
            continue
        if not with_t and share < 0.3:
            continue
        ids = [index.ids[j] for j in where[n]]
        cands.append({"name": n, "count": c, "share": round(share, 3), "lift": round(lift, 2), "cited_ids": ids[:5],
                      "kind": "direct"})
    cands.sort(key=lambda d: (-d["count"], -d["lift"], d["name"]))
    cands = cands[:5]
    strength = "none"
    if cands:
        top = cands[0]
        strength = "strong" if top["count"] >= 5 and top["share"] >= 0.25 else ("moderate" if top["count"] >= 3 else "weak")
    competing = len(cands) > 1 and cands[1]["count"] >= 0.8 * cands[0]["count"] and cands[0]["share"] < 0.6
    hints = []
    key = _role_key(T, HINTS)
    if key:
        for h in HINTS[key]:
            hn = normalize_ingredient(h)
            if (len(index.name_index.get(hn, ())) < 20 or hn in orig_names or related(T, hn)
                    or any(m.search(hn) for m in restr_matchers) or any(d["name"] == hn for d in cands)):
                continue
            fam = [index.ids[j] for _, j in without_t if hn in index.names[j]]
            hints.append({"name": hn, "kind": "indirect", "family_count": len(fam), "cited_ids": fam[:5],
                          "dataset_recipes_with_ingredient": len(index.name_index[hn])})
    hints.sort(key=lambda d: (-d["family_count"], d["name"]))
    hints = hints[:3]
    omit = None
    if len(without_t) >= 3:
        omit = {"kind": "omit", "count": len(without_t), "of": len(neigh), "cited_ids": [index.ids[j] for _, j in without_t[:5]]}
    chosen, level = None, "none"
    if cands:
        chosen, level = cands[0], strength
    else:
        fam_hints = [h for h in hints if h["family_count"] >= 1]
        if fam_hints:
            chosen, level = fam_hints[0], "indirect (weak)"
        elif omit:
            chosen, level = omit, "weak"
    role = _role_key(T, ROLES)
    role_txt = ROLES[role] if role else None
    notes = []
    if len(neigh) < 5:
        notes.append(f"Only {len(neigh)} similar recipe(s) were found, so any pattern is based on very little data.")
    if competing:
        notes.append("Several candidate substitutes have comparable support: the data does not single out one (conflicting/competing evidence).")
    if not cands:
        notes.append("No ingredient is consistently used in place of this one in similar recipes.")
    if chosen and chosen["kind"] == "indirect":
        notes.append("Chosen option is INDIRECT: suggested by general culinary knowledge; the dataset only shows the ingredient exists/appears in related recipes.")
    if chosen and chosen["kind"] == "omit":
        notes.append("Omission is supported only by the fact that some similar recipes do not use this ingredient; whether the result works is not verified.")
    prep = []
    if chosen and chosen["kind"] in ("direct", "indirect"):
        for rid in chosen["cited_ids"][:2]:
            ex = _step_example(index, rid, chosen["name"])
            if ex:
                prep.append(ex)
    slot = {"target": T, "lines": [lines[i] for i in line_idx], "line_indices": line_idx, "neighbors": len(neigh),
            "with_target": len(with_t), "without_target": len(without_t), "candidates": cands, "hints": hints, "omit": omit,
            "strength": strength, "competing": competing, "chosen": chosen, "evidence_level": level,
            "role": role_txt[0] if role_txt else None,
            "prediction": (f"Prediction (general culinary knowledge, NOT verified by this dataset): '{T}' typically provides {role_txt[0]}; "
                           f"changing it {role_txt[1]}.") if role_txt else
                          f"No role information is available for '{T}'; effects on taste/texture cannot be predicted from the dataset.",
            "prep_patterns": prep, "notes": notes}
    kind_map = {"direct": "candidate", "indirect": "general_knowledge", "omit": "omit"}
    if chosen:
        slot["evidence_class"] = evidence_class(slot, kind_map[chosen["kind"]], chosen.get("name"))
        if chosen["kind"] == "direct":
            slot["why"] = (f"{chosen['count']} of {len(without_t)} similar recipes that do not use '{T}' use '{chosen['name']}' "
                           f"(lift {chosen['lift']} vs similar recipes that do use '{T}'). This is co-occurrence in comparable recipes, not proof it works here.")
        elif chosen["kind"] == "indirect":
            slot["why"] = (f"'{chosen['name']}' is a commonly used alternative in general culinary practice; {chosen['family_count']} similar recipe(s) in the dataset use it "
                           "without the target, so support is indirect.")
        else:
            slot["why"] = f"{chosen['count']} of {chosen['of']} similar recipes do not use '{T}' at all, so omitting it is at least not unprecedented."
    else:
        slot["evidence_class"] = "Insufficient"
        slot["why"] = "The dataset does not provide adequate evidence for a replacement."
    return slot


def propose_substitution(index, recipe_id: str, targets, restrictions=(), max_slots=4) -> dict:
    if recipe_id not in index.id_to_idx:
        return {"error": f"Recipe ID {recipe_id} not found in the database."}
    idx0 = index.id_to_idx[recipe_id]
    full = index.full(recipe_id)
    lines = full["ingredients"]
    steps = full["instructions_steps"]
    tm = [get_matcher(t) for t in targets]
    rm = [get_matcher(r) for r in restrictions]
    all_m = list(tm)
    for m in rm:
        if all(m.label != x.label for x in all_m):
            all_m.append(m)
    flagged: dict[int, list] = {}
    for li, l in enumerate(lines):
        if not normalize_ingredient(l):
            continue
        for m in all_m:
            hit = m.search(l.lower())
            if hit:
                flagged.setdefault(li, []).append((m.label, hit))
    base = {"recipe": {k: full[k] for k in ("id", "title", "source_file", "source_key", "ingredients", "instructions_steps", "step_count")},
            "targets": list(targets), "restrictions": [m.label for m in rm], "disclaimer": DISCLAIMER}
    if not flagged:
        base["error"] = ("No ingredient line in this recipe matches " + ", ".join(f"'{t}'" for t in targets) +
                         (" or the active restrictions" if rm else "") + ". Nothing to substitute.")
        return base
    groups: dict[str, list] = {}
    for li in sorted(flagged):
        groups.setdefault(normalize_ingredient(lines[li]), []).append(li)
    slot_names = list(groups)[:max_slots]
    orig_names = set(index.names[idx0])
    restr_for_candidates = all_m
    slots = [analyse_slot(index, idx0, T, restr_for_candidates, orig_names, groups[T], lines) for T in slot_names]
    notes = []
    if len(groups) > max_slots:
        notes.append(f"{len(groups) - max_slots} further flagged ingredient(s) were not analysed (limit {max_slots}).")
    line_slot = {li: s for s in slots for li in s["line_indices"]}
    new_ing = []
    for li, l in enumerate(lines):
        s = line_slot.get(li)
        if s is None:
            new_ing.append({"text": l, "status": "unresolved — still matches restriction/target" if li in flagged else "unchanged"})
        elif s["chosen"] is None:
            new_ing.append({"text": l, "status": "unresolved — no supported substitute"})
        elif s["chosen"]["kind"] == "omit":
            new_ing.append({"text": l, "status": "removed (omitted)"})
        else:
            new_ing.append({"text": f"{s['chosen']['name']} — amount not verified; start from the original amount and adjust",
                            "status": "substituted", "replaces": l})
    new_steps = []
    for n, st in enumerate(steps, 1):
        cur, changed, review = st, False, False
        for s in slots:
            if not s["chosen"]:
                continue
            head = re.escape(s["target"].split()[-1])
            rx = re.compile(rf"\b{head}(?:s|es)?\b", re.I)
            if rx.search(cur):
                if s["chosen"]["kind"] == "omit":
                    review = True
                else:
                    cur, changed = rx.sub(s["chosen"]["name"], cur), True
        status = "edited (automatic word replacement — review)" if changed else ("review: mentions an omitted ingredient" if review else "unchanged")
        new_steps.append({"n": n, "text": cur, "status": status, "original": st if changed else None})
    remaining = []
    for item in new_ing:
        if item["status"].startswith("removed"):
            continue
        low = item["text"].lower().split(" — ")[0]
        for m in all_m:
            if m.search(low):
                remaining.append({"line": item["text"], "matches": m.label})
                break
    any_choice = any(s["chosen"] for s in slots)
    base.update({"slots": slots, "revised_ingredients": new_ing, "revised_instructions": new_steps,
                 "remaining_violations": remaining, "revision_supported": any_choice, "notes": notes,
                 "overall_evidence": (min((x["evidence_class"] for x in slots), key=["Insufficient", "Weak or conflicting", "Indirect", "Direct"].index) if slots else "Insufficient")})
    return base


def general_substitution(index, term: str, restrictions=(), llm_options=None) -> dict:
    """Ingredient-level answer when no recipe is named: INDIRECT evidence only (ingredient presence + related recipes)."""
    T = normalize_ingredient(term)
    rm = [get_matcher(r) for r in restrictions] + [get_matcher(term)]
    with_t = sorted(index.contains(T))[:3000]
    generic = {"recipe", "easy", "best", "simple", "quick", "classic", "homemade", "with", "and", "the", "style"}
    tc = Counter(t for i in with_t for t in index.title_tokens[i] if t not in generic and len(t) > 2)
    top_tokens = {t for t, _ in tc.most_common(6)}
    key = _role_key(T, HINTS)
    names = list(HINTS.get(key, [])) if key else []
    opts, seen = [], set()
    for src, lst in (("curated", names), ("ai", [o["name"] for o in (llm_options or [])])):
        for h in lst:
            hn = normalize_ingredient(h)
            if not hn or hn in seen or related(T, hn) or any(m.search(hn) for m in rm):
                continue
            n_rec = len(index.name_index.get(hn, ()))
            if n_rec < 20:
                continue
            seen.add(hn)
            ex = [index.ids[j] for j in index.name_index[hn] if index.title_tokens[j] & top_tokens and not any(related(T, n) for n in index.names[j])][:3]
            opts.append({"name": hn, "dataset_recipes": n_rec, "example_ids": ex, "source": src})
    role = ROLES.get(key) if key else None
    return {"term": T, "n_recipes_with": len(index.contains(T)), "options": opts[:5], "evidence_class": "Indirect" if opts else "Insufficient",
            "role": role[0] if role else None, "prediction": (f"Prediction (general culinary knowledge, NOT verified by this dataset): '{T}' typically provides {role[0]}; replacing it {role[1]}.") if role else None,
            "restrictions": [m.label for m in rm[:-1]], "disclaimer": DISCLAIMER,
            "notes": ["No recipe was named, so only INDIRECT evidence is available: the alternatives exist in the dataset and appear in recipes related to those that use the ingredient. Name a recipe (e.g. “in the penne pasta recipe”) for a full, evidence-based adaptation."]}
