"""HTML/Streamlit rendering helpers. NOTE: the pandas-backed Streamlit data widgets are deliberately not used — they can import pandas
(blocked by some Windows Application Control policies). Tables are rendered as plain HTML instead."""
from __future__ import annotations
import html

import streamlit as st


def e(x) -> str:
    return html.escape(str(x), quote=True)


def md_escape(s) -> str:
    return str(s).replace("$", "\\$").replace("<", "&lt;")


def chip(t, cls="") -> str:
    return f'<span class="chip {cls}">{e(t)}</span>'


def chips(items, cls="", empty="none") -> str:
    return '<div class="chips">' + ("".join(chip(i, cls) for i in items) if items else chip(empty, "mute")) + "</div>"


def table(rows, cols) -> str:
    """rows: list of dicts; cols: list of (key, header)."""
    head = "".join(f"<th>{e(h)}</th>" for _, h in cols)
    body = "".join("<tr>" + "".join(f"<td>{e(r.get(k, ''))}</td>" for k, _ in cols) + "</tr>" for r in rows)
    return f'<div class="tbl-wrap"><table class="sb-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def tiles(pairs) -> str:
    return '<div class="tiles">' + "".join(f'<div class="tile"><b>{e(v)}</b><span>{e(k)}</span></div>' for k, v in pairs) + "</div>"


def banner(text, kind="info"):
    st.markdown(f'<div class="banner {kind}">{text}</div>', unsafe_allow_html=True)


def card_head(title, rid, meta=(), score=None, score_label="match score") -> str:
    right = f'<div class="rc-score"><b>{e(score)}</b><span>{e(score_label)}</span></div>' if score is not None else ""
    return ('<div class="rc-top"><div class="rc-badge">' + e((title or "?")[:1].upper()) + '</div><div><div class="rc-title">' + e(title) +
            '</div><div class="rc-id">ID ' + e(rid) + "</div></div>" + right + "</div>" + chips(meta, "", "") )


def original_recipe(full):
    st.markdown("**Ingredients (original)**")
    st.markdown("\n".join("- " + md_escape(l) for l in full["ingredients"]))
    st.markdown("**Instructions (original)**")
    st.markdown("\n".join(f"{n}. {md_escape(s)}" for n, s in enumerate(full["instructions_steps"], 1)) or "_No instructions in the source record._")
    st.caption(f"Source: {full['source_file']} · key {full['source_key']} · step-count method: {full['step_method']}")


def recipe_card(index, item):
    with st.container(border=True):
        meta = [f"{item['ingredient_lines']} ingredient lines", f"{item['steps'] or 'unknown'} steps"]
        st.markdown(card_head(item["title"], item["id"], meta), unsafe_allow_html=True)
        for reason in item.get("why") or []:
            st.markdown(f'<div class="note">🔎 {e(reason)}</div>', unsafe_allow_html=True)
        if item.get("restriction_check"):
            st.markdown(f'<div class="note">✓ {e(item["restriction_check"])}</div>', unsafe_allow_html=True)
        with st.expander("Original ingredients & instructions"):
            original_recipe(index.full(item["id"]))


def stats_block(r):
    pairs = [("Recipes analysed", f"{r['n_analyzed']:,}"), ("Denominator", f"{r['denominator']:,}")]
    if r["kind"] == "count":
        pairs.append(("Share", f"{r['percentage']}%"))
        pairs.append(("Count", f"{r['count']:,}"))
    st.markdown(tiles(pairs), unsafe_allow_html=True)
    if r["kind"] == "count":
        st.markdown(f'<div class="note">Strict exact-name count: {r["exact_name_count"]:,} ({r["exact_name_percentage"]}%). '
                    f'Percent of the whole dataset ({r["whole_dataset_size"]:,}): {r["percentage_of_whole_dataset"]}%.'
                    + (" Example IDs: " + e(", ".join(r["sample_ids"])) if r["sample_ids"] else "") + "</div>", unsafe_allow_html=True)
    elif r["kind"] == "top_ingredients":
        st.markdown(table(r["rows"], [("rank", "#"), ("ingredient", "Ingredient"), ("recipes", "Recipes"), ("percent", "% of recipes")]), unsafe_allow_html=True)
    elif r["kind"] == "rank_steps":
        st.markdown(table(r["rows"], [("rank", "#"), ("id", "Recipe ID"), ("title", "Title"), ("steps", "Steps")]), unsafe_allow_html=True)
    elif r["kind"] == "ingredient_stats" and r["n_analyzed"]:
        st.markdown(tiles([("Mean", r["mean"]), ("Median", r["median"]), ("Min", r["min"]), ("Max", r["max"])]), unsafe_allow_html=True)
    with st.expander("Filters, normalisation & calculation", expanded=True):
        st.markdown("**Filters applied:** " + md_escape("; ".join(r["filters"])))
        st.markdown("**Normalisation / counting rule:** " + md_escape(r["normalization"]))
        st.markdown("**Calculation:** " + md_escape(r["method"]))
        for n in r["notes"]:
            st.caption(md_escape(n))


def match_card(index, r, ineligible=False):
    with st.container(border=True):
        meta = [f"{r['steps'] or 'unknown'} steps", f"{r['ingredient_names']} ingredient names", f"uses {r['n_used']} of yours"]
        st.markdown(card_head(r["title"], r["id"], meta, r["score"], "score / 100"), unsafe_allow_html=True)
        if ineligible:
            st.markdown('<div class="banner err"><b>INELIGIBLE</b> — violates: ' + e("; ".join(r["violations"])) + "</div>", unsafe_allow_html=True)
        elif r["partial_only"]:
            st.markdown('<div class="banner warn">Partial match: uses fewer of your ingredients than the other results.</div>', unsafe_allow_html=True)
        lab = lambda m: m["recipe_ingredient"] + (f" (variant of {m['your_ingredient']})" if m["match"] == "variant" else "") + (" · staple assumed" if m["pantry_staple_assumed"] else "")
        st.markdown('<div class="lbl">Matched</div>' + chips([lab(m) for m in r["matched"]], "ok"), unsafe_allow_html=True)
        st.markdown(f'<div class="lbl">Missing / additional required ({len(r["missing"])})</div>' + chips(r["missing"], "warn"), unsafe_allow_html=True)
        st.markdown('<div class="lbl">Your ingredients not used</div>' + chips(r["unused_available"], "mute"), unsafe_allow_html=True)
        c = r["components"]
        w = r["weights"]
        st.markdown('<div class="note"><b>Why this rank:</b> ' + e(r["reason"]) + "</div>", unsafe_allow_html=True)
        with st.expander("Score breakdown (reproducible) & original recipe"):
            rows = [{"c": "Usage", "v": c["usage"], "w": w["usage"], "p": round(c["usage"] * w["usage"], 2)},
                    {"c": "Coverage", "v": c["coverage"], "w": w["coverage"], "p": round(c["coverage"] * w["coverage"], 2)},
                    {"c": "Missing", "v": c["missing"], "w": w["missing"], "p": round(c["missing"] * w["missing"], 2)},
                    {"c": "Steps", "v": c["steps"], "w": w["steps"], "p": round(c["steps"] * w["steps"], 2)}]
            st.markdown(table(rows, [("c", "Component"), ("v", "Value (0-100)"), ("w", "Weight"), ("p", "Contribution")]), unsafe_allow_html=True)
            original_recipe(index.full(r["id"]))


def matches_block(index, res):
    st.markdown('<div class="note">Name-level match; quantity sufficiency not verified.</div>', unsafe_allow_html=True)
    for n in res["notes"][1:]:
        st.markdown(f'<div class="note">{e(n)}</div>', unsafe_allow_html=True)
    for r in res["eligible"]:
        match_card(index, r)
    if res["alternatives"]:
        banner("<b>Ineligible alternatives</b> — shown only for reference. Each violates a hard constraint you set and is <b>not</b> a recommendation.", "warn")
        for r in res["alternatives"]:
            match_card(index, r, True)
    with st.expander("Scoring method & result funnel"):
        st.markdown(md_escape(res["formula"]))
        st.markdown(f"Preference weights used (usage / coverage / missing / steps): **{res['weights']['usage']} / {res['weights']['coverage']} / {res['weights']['missing']} / {res['weights']['steps']}** — {md_escape(res['goal_label'])}.")
        if res["assume_staples"]:
            st.markdown("Pantry staples assumed available: " + ", ".join(res["staples"]) + ". (Turn off by saying “don't assume pantry staples”.)")
        st.markdown(f"Funnel: {res['n_candidates']:,} recipes share ≥1 of your ingredients → {res['excluded_by_steps']:,} over the step limit/unknown steps, {res['excluded_by_exclusion']:,} contain an excluded ingredient → **{res['n_eligible']:,} eligible**.")


def _evidence_banner(cls):
    kind = {"Direct": "ok", "Indirect": "warn", "Weak or conflicting": "warn", "Insufficient": "err"}.get(cls, "info")
    expl = {"Direct": "similar local recipes without the ingredient use this alternative",
            "Indirect": "related recipes/ingredient combinations suggest it, but do not directly establish it works here",
            "Weak or conflicting": "the dataset does not adequately support a confident recommendation",
            "Insufficient": "no usable evidence"}.get(cls, "")
    return f'<div class="banner {kind}"><b>Evidence: {e(cls)}</b> — {e(expl)}.</div>'


def sub_block(index, res):
    rec = res["recipe"]
    st.markdown('<div class="banner warn">' + e(res["disclaimer"]) + "</div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(card_head(rec["title"], rec["id"], [f"{len(rec['ingredients'])} ingredient lines", f"{rec['step_count']} steps"]), unsafe_allow_html=True)
        with st.expander("ORIGINAL recipe (unchanged)"):
            original_recipe(index.full(rec["id"]))
    llm = res.get("llm", {})
    if llm.get("reason"):
        st.markdown(f'<div class="note">ℹ️ {e(llm["reason"])}</div>', unsafe_allow_html=True)
    for k, s in enumerate(res["slots"]):
        with st.container(border=True):
            st.markdown(f'<div class="lbl">1 · Unavailable / restricted ingredient</div>' + chips([s["target"]] + s["lines"][:2], "err"), unsafe_allow_html=True)
            use_ai = llm.get("used") and k == 0
            ec = llm["evidence_class"] if use_ai else s["evidence_class"]
            ch = s["chosen"]
            repl = llm["replacement"] if use_ai else (ch.get("name", "omit it") if ch else "none supported")
            st.markdown('<div class="lbl">2 · Suggested replacement</div>' + chips([repl], "ok" if ch or use_ai else "mute"), unsafe_allow_html=True)
            why = llm.get("why") if use_ai else s["why"]
            st.markdown(f'<div class="lbl">3 · Why it may work</div><div class="note" style="font-size:.92rem;color:#29221D">{e(why)}</div>', unsafe_allow_html=True)
            if use_ai and llm.get("role"):
                st.markdown(f'<div class="note"><b>Role of the ingredient (AI reasoning):</b> {e(llm["role"])}</div>', unsafe_allow_html=True)
            elif s["role"]:
                st.markdown(f'<div class="note"><b>Likely role:</b> {e(s["role"])} <i>(general culinary knowledge)</i></div>', unsafe_allow_html=True)
            ids = llm["cited_ids"] if use_ai else (ch["cited_ids"] if ch else [])
            st.markdown('<div class="lbl">4 · Supporting local recipe IDs (verified in the database)</div>' + chips(ids, "", "none cited"), unsafe_allow_html=True)
            st.markdown(f"{_evidence_banner(ec)}", unsafe_allow_html=True)
            st.markdown(f'<div class="note">Similar recipes found: {s["neighbors"]} ({s["with_target"]} use the ingredient, {s["without_target"]} do not).</div>', unsafe_allow_html=True)
            if s["candidates"]:
                st.markdown(table([{"n": c["name"], "c": c["count"], "sh": c["share"], "l": c["lift"], "i": ", ".join(c["cited_ids"][:3])} for c in s["candidates"]],
                                  [("n", "Alternative seen in similar recipes"), ("c", "Recipes"), ("sh", "Share"), ("l", "Lift"), ("i", "Example IDs")]), unsafe_allow_html=True)
            if s["hints"]:
                st.markdown('<div class="lbl">Indirect ideas (general knowledge; ingredient exists in dataset)</div>' + table(
                    [{"n": h["name"], "f": h["family_count"], "d": h["dataset_recipes_with_ingredient"], "i": ", ".join(h["cited_ids"][:3])} for h in s["hints"]],
                    [("n", "Idea"), ("f", "Similar recipes using it"), ("d", "All dataset recipes"), ("i", "Example IDs")]), unsafe_allow_html=True)
            if use_ai and llm.get("effects"):
                ef = llm["effects"]
                st.markdown('<div class="lbl">8 · Expected changes — PREDICTIONS, not verified</div>' + table(
                    [{"a": "Flavour", "b": ef["flavour"]}, {"a": "Texture", "b": ef["texture"]}, {"a": "Preparation", "b": ef["preparation"]}], [("a", ""), ("b", "Prediction")]), unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="lbl">8 · Expected changes</div><div class="note">{e(s["prediction"])}</div>', unsafe_allow_html=True)
            unc = (llm.get("uncertainties", []) + llm.get("notes", [])) if use_ai else []
            unc += s["notes"]
            if unc:
                st.markdown('<div class="lbl">9 · Uncertainties & limitations</div>' + "".join(f'<div class="note">• {e(u)}</div>' for u in unc), unsafe_allow_html=True)
            for p in s["prep_patterns"]:
                st.markdown(f'<div class="note">Preparation example from <code>{e(p["recipe_id"])}</code>: “{e(p["step"])}”</div>', unsafe_allow_html=True)
    if res["revision_supported"]:
        st.markdown('<div class="sec" style="font-size:1.2rem">PROPOSED adaptation <span class="chip warn">not part of the original</span></div>', unsafe_allow_html=True)
        st.markdown("**Updated ingredient list**")
        lines = []
        for it in res["revised_ingredients"]:
            t = md_escape(it["text"])
            if it["status"] in ("substituted", "ai-revised"):
                lines.append(f"- 🔁 **{t}**" + (f"  \n  <small>replaces: {md_escape(it['replaces'])}</small>" if it.get("replaces") else ""))
            elif it["status"].startswith("removed"):
                lines.append(f"- ~~{t}~~ — removed")
            else:
                lines.append(f"- {t}" + ("" if it["status"] == "unchanged" else f" _[{it['status']}]_"))
        st.markdown("\n".join(lines), unsafe_allow_html=True)
        st.markdown("**Revised instructions**")
        st.markdown("\n".join(f"{x['n']}. {md_escape(x['text'])}" + ("" if x["status"] == "unchanged" else f" _[{x['status']}]_") for x in res["revised_instructions"]))
        if res["remaining_violations"]:
            banner("Lines still matching a restriction/target: " + e("; ".join(v["line"] for v in res["remaining_violations"])), "err")
        elif not llm.get("used"):
            st.markdown('<div class="note">Instruction edits are automatic word replacement — read them before cooking.</div>', unsafe_allow_html=True)
    for n in res["notes"]:
        st.caption(n)


def general_sub_block(res):
    st.markdown('<div class="banner warn">' + e(res["disclaimer"]) + "</div>", unsafe_allow_html=True)
    st.markdown(_evidence_banner(res["evidence_class"]), unsafe_allow_html=True)
    if res.get("role"):
        st.markdown(f'<div class="note"><b>Likely role of {e(res["term"])}:</b> {e(res["role"])} <i>(general culinary knowledge)</i></div>', unsafe_allow_html=True)
    if res["options"]:
        st.markdown(table([{"n": o["name"], "d": f"{o['dataset_recipes']:,}", "i": ", ".join(o["example_ids"]) or "—", "s": ("AI-suggested" if o["source"] == "ai" else "common alternative")} for o in res["options"]],
                          [("n", "Alternative"), ("d", "Dataset recipes using it"), ("i", "Related example IDs"), ("s", "Origin")]), unsafe_allow_html=True)
    if res.get("prediction"):
        st.markdown(f'<div class="note">{e(res["prediction"])}</div>', unsafe_allow_html=True)
    for n in res["notes"]:
        st.caption(n)


def render_blocks(index, blocks):
    for b in blocks:
        t = b["type"]
        if t == "recipes":
            for it in b["items"]:
                recipe_card(index, it)
        elif t == "details":
            with st.container(border=True):
                st.markdown(card_head(b["recipe"]["title"], b["recipe"]["id"], [f"{b['recipe']['step_count']} steps"]), unsafe_allow_html=True)
                original_recipe(b["recipe"])
        elif t == "stats":
            stats_block(b["result"])
        elif t == "matches":
            matches_block(index, b["result"])
        elif t == "substitution":
            sub_block(index, b["result"])
        elif t == "general_sub":
            general_sub_block(b["result"])
