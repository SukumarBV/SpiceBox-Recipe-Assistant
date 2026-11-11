"""LLM reasoning for substitutions, GROUNDED in retrieved local evidence and VALIDATED by code.

One compact request per adaptation. The model may only cite recipe IDs that we retrieved from the database;
its proposed ingredient list is checked against hard restrictions; the evidence class is computed by us,
never taken from the model. Any failure falls back to the deterministic result.
"""
from __future__ import annotations
import json

from .ingredient_normalizer import get_matcher, normalize_ingredient
from .llm import LLMError

SYSTEM = ("You are a careful culinary adviser. You adapt ONE recipe using ONLY the evidence given. Reason about the role of the "
          "unavailable ingredient (flavour, moisture, binding, thickening, texture, fat, acidity, structure, colour, cooking behaviour); "
          "do not assume one-to-one swaps. Never cite a recipe ID that is not in evidence_recipes. Never claim medical or allergy safety. "
          "Label effects as predictions. Output ONLY JSON.")
OUT_HINT = ('{"role":str,"replacement":str,"replacement_kind":"candidate|general_knowledge|omit","why":str,"cited_ids":[ids from evidence_recipes],'
            '"ingredients":[complete revised ingredient lines],"steps":[complete revised instruction steps],'
            '"effects":{"flavour":str,"texture":str,"preparation":str},"uncertainties":[str]}')


def evidence_class(slot, kind: str, name: str | None) -> str:
    if kind == "omit":
        return "Weak or conflicting"
    if kind == "candidate":
        c = next((x for x in slot["candidates"] if x["name"] == name), None)
        if c and not slot["competing"] and c["count"] >= 3:
            return "Direct"
        return "Weak or conflicting"
    h = next((x for x in slot["hints"] if x["name"] == name), None)
    return "Indirect" if h and h["family_count"] >= 1 else "Weak or conflicting"


def _evidence_ids(slot):
    ids = []
    for grp in (slot["candidates"][:3], slot["hints"][:2]):
        for c in grp:
            for i in c["cited_ids"][:3]:
                if i not in ids:
                    ids.append(i)
    if slot["omit"]:
        for i in slot["omit"]["cited_ids"][:2]:
            if i not in ids:
                ids.append(i)
    return ids[:6]


def build_prompt(index, result, slot, restrictions):
    rec = result["recipe"]
    ev = [{"id": i, "title": index.titles[index.id_to_idx[i]], "ingredients": list(index.names[index.id_to_idx[i]])[:14]} for i in _evidence_ids(slot)]
    payload = {
        "recipe": {"id": rec["id"], "title": rec["title"], "ingredients": [l[:80] for l in rec["ingredients"][:25]],
                   "steps": [s[:160] for s in rec["instructions_steps"][:12]]},
        "unavailable_ingredient": slot["target"], "hard_restrictions": restrictions,
        "dataset_evidence": {"similar_recipes": slot["neighbors"], "using_it": slot["with_target"], "without_it": slot["without_target"],
                             "candidates": [{"name": c["name"], "recipes": c["count"], "share": c["share"]} for c in slot["candidates"][:4]],
                             "general_ideas": [h["name"] for h in slot["hints"][:3]],
                             "omit_support": slot["omit"]["count"] if slot["omit"] else 0},
        "evidence_recipes": ev,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\nReturn JSON exactly like: " + OUT_HINT, [e["id"] for e in ev]


def refine(llm, index, result, restrictions, primary_target=None):
    """Mutates and returns `result`. Returns result unchanged (with result['llm']={'used':False,'reason':…}) on any problem."""
    result["llm"] = {"used": False, "reason": ""}
    slots = result.get("slots") or []
    slot = next((s for s in slots if primary_target and (s["target"] in primary_target or primary_target in s["target"])), slots[0] if slots else None)
    if not slot or not (slot["candidates"] or slot["hints"] or slot["omit"]):
        result["llm"]["reason"] = "No local evidence was found, so no AI reasoning was requested (nothing to ground it on)."
        return result
    if not (llm and llm.available):
        result["llm"]["reason"] = "AI not configured or paused; deterministic evidence-based result shown."
        return result
    prompt, ev_ids = build_prompt(index, result, slot, result["restrictions"])
    try:
        out = llm.complete_json(SYSTEM, prompt, max_tokens=900)
    except LLMError as e:
        result["llm"]["reason"] = f"AI unavailable ({e}); deterministic result shown."
        return result
    notes = []
    kind = out.get("replacement_kind")
    name = normalize_ingredient(str(out.get("replacement", ""))) if kind != "omit" else None
    ings, steps = out.get("ingredients"), out.get("steps")
    if kind not in ("candidate", "general_knowledge", "omit") or (kind != "omit" and not name):
        result["llm"]["reason"] = "AI output rejected: malformed replacement; deterministic result shown."
        return result
    if not (isinstance(ings, list) and 1 <= len(ings) <= 60 and all(isinstance(x, str) and 0 < len(x) <= 220 for x in ings)
            and isinstance(steps, list) and 1 <= len(steps) <= 40 and all(isinstance(x, str) and x.strip() for x in steps)):
        result["llm"]["reason"] = "AI output rejected: ingredient/step lists malformed; deterministic result shown."
        return result
    if kind == "candidate" and name not in [c["name"] for c in slot["candidates"]]:
        kind = "general_knowledge"
        notes.append("The AI named a replacement that is not among the evidence-derived candidates; it is treated as general knowledge.")
    all_m = [get_matcher(t) for t in result["targets"]] + [get_matcher(r) for r in restrictions]
    if kind != "omit" and any(m.search(name) for m in all_m):
        result["llm"]["reason"] = "AI output rejected: proposed replacement violates a hard restriction/target; deterministic result shown."
        return result
    if kind == "general_knowledge" and len(index.name_index.get(name, ())) < 20:
        notes.append(f"'{name}' is rare or absent in this dataset, so the dataset gives essentially no evidence about it.")
    cited = [i for i in (out.get("cited_ids") or []) if isinstance(i, str)]
    good = [i for i in cited if i in ev_ids and i in index.id_to_idx]
    if len(good) != len(cited):
        notes.append(f"{len(cited) - len(good)} recipe ID(s) cited by the AI were not in the retrieved evidence and were discarded.")
    orig_ing, orig_steps = set(result["recipe"]["ingredients"]), set(result["recipe"]["instructions_steps"])
    result["deterministic"] = {"revised_ingredients": result["revised_ingredients"], "revised_instructions": result["revised_instructions"]}
    result["revised_ingredients"] = [{"text": l, "status": "unchanged" if l in orig_ing else "ai-revised"} for l in ings]
    result["revised_instructions"] = [{"n": n, "text": s, "status": "unchanged" if s in orig_steps else "ai-revised", "original": None} for n, s in enumerate(steps, 1)]
    remaining = []
    for it in result["revised_ingredients"]:
        for m in all_m:
            if m.search(it["text"].lower()):
                remaining.append({"line": it["text"], "matches": m.label})
                break
    result["remaining_violations"] = remaining
    result["revision_supported"] = True
    ec = evidence_class(slot, kind, name)
    eff = out.get("effects") if isinstance(out.get("effects"), dict) else {}
    result["llm"] = {"used": True, "reason": "", "role": str(out.get("role", ""))[:300], "replacement": name or "omit", "kind": kind,
                     "why": str(out.get("why", ""))[:600], "cited_ids": good, "evidence_class": ec,
                     "effects": {k: str(eff.get(k, ""))[:300] for k in ("flavour", "texture", "preparation")},
                     "uncertainties": [str(u)[:200] for u in (out.get("uncertainties") or [])][:5], "notes": notes}
    result["overall_evidence"] = ec
    return result


def general_with_llm(llm, index, term, options, restrictions):
    """Ingredient-level (no recipe) suggestions: LLM proposes, code verifies presence in dataset + restrictions."""
    if not (llm and llm.available):
        return []
    user = json.dumps({"ingredient": term, "hard_restrictions": restrictions, "dataset_options": [o["name"] for o in options]}, separators=(",", ":")) + \
        '\nSuggest up to 4 common alternatives and the ingredient role. JSON: {"role":str,"alternatives":[{"name":str,"note":str}]}'
    try:
        out = llm.complete_json(SYSTEM, user, max_tokens=300)
    except LLMError:
        return []
    ms = [get_matcher(r) for r in restrictions] + [get_matcher(term)]
    res = []
    for a in (out.get("alternatives") or [])[:4]:
        if not isinstance(a, dict):
            continue
        n = normalize_ingredient(str(a.get("name", "")))
        if n and not any(m.search(n) for m in ms) and len(index.name_index.get(n, ())) >= 20:
            res.append({"name": n, "note": str(a.get("note", ""))[:200], "role": str(out.get("role", ""))[:200]})
    return res
