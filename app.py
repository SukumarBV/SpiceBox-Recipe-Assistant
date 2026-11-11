"""SpiceBox — AI Recipe Assistant (Streamlit). Works without any API key; an LLM is an optional enhancement."""
from __future__ import annotations
import streamlit as st

from core import config
from core.conversation import ConversationState, Response, handle_message
from core.database import RecipeIndex, db_exists
from core.db_fetch import download_db, get_url
from core.ingredient_matcher import GOAL_LABELS
from core.llm import LLMClient, load_config
from ui.components import chip, chips, e, md_escape, render_blocks
from ui.style import CSS

st.set_page_config(page_title="SpiceBox — AI Recipe Assistant", page_icon="🍲", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

LOGO = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 11h16a8 8 0 0 1-16 0z"/><path d="M9 7c0-1.5 1.5-1.5 1.5-3M14 7c0-1.5 1.5-1.5 1.5-3"/></svg>')


def _secrets():
    try:
        return st.secrets
    except Exception:
        return None


@st.cache_resource
def get_llm() -> LLMClient:
    return LLMClient(load_config(_secrets()))


@st.cache_resource(show_spinner="Loading the recipe collection…")
def get_index():
    return RecipeIndex.from_db(config.DB_PATH)


def setup_screen():
    st.markdown('<div class="sb-header"><div class="sb-brand"><div class="sb-logo">' + LOGO + '</div><div>SpiceBox<small>AI Recipe Assistant</small></div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">One-time setup needed</div>', unsafe_allow_html=True)
    st.error("The recipe database was not found, so no recipes can be shown (nothing is ever made up).")
    st.markdown(f"""
Put the three dataset files in `data/raw/`, then run in PowerShell from the project folder:
```
python scripts/inspect_dataset.py
python scripts/build_database.py
streamlit run app.py
```
Expected database: `{config.DB_PATH}`. Hosting without local files: see the README section *Deployment* (`DB_URL`).
""")
    st.stop()


if not db_exists(config.DB_PATH):
    url = get_url(_secrets())
    if url:
        try:
            with st.spinner("Downloading the recipe database (first start only)…"):
                download_db(url, config.DB_PATH)
            st.rerun()
        except Exception as ex:
            st.error(f"Could not download the database ({type(ex).__name__}). Check DB_URL.")
    setup_screen()

index = get_index()
llm = get_llm()
if "conv" not in st.session_state:
    st.session_state.conv = ConversationState()
    st.session_state.messages = []
conv: ConversationState = st.session_state.conv


def process(msg: str):
    st.session_state.messages.append({"role": "user", "content": msg, "blocks": []})
    try:
        r = handle_message(msg, conv, index, llm)
    except Exception as ex:  # keep the UI alive and be honest
        r = Response(f"Sorry — something went wrong handling that ({type(ex).__name__}). Your session state is unchanged where possible.")
    st.session_state.messages.append({"role": "assistant", "content": r.text, "blocks": r.blocks})


def ask(msg: str):
    st.session_state.pending = msg
    st.rerun()


def do_reset():
    st.session_state.conv = ConversationState()
    st.session_state.messages = []


pending = st.session_state.pop("pending", None)
if pending:
    with st.spinner("Searching the recipe collection…"):
        process(pending)

# ------------------------------------------------------------------ header
hc = st.columns([5, 1.2], vertical_alignment="center")
dot = "on" if llm.available else ("pause" if llm.configured else "")
hc[0].markdown(
    '<div class="sb-header"><div class="sb-brand"><div class="sb-logo">' + LOGO + '</div><div>SpiceBox<small>AI Recipe Assistant</small></div></div>'
    '<div class="sb-nav"><a href="#hero">Search</a><a href="#tools">Cook &amp; Explore</a><a href="#chat">Conversation</a><a href="#how">How it works</a></div>'
    f'<div class="sb-status"><span class="sb-dot {dot}"></span>{e(llm.status())}</div></div>', unsafe_allow_html=True)
hc[1].button("Reset conversation", on_click=do_reset, key="reset_top", use_container_width=True)

# ------------------------------------------------------------------ hero
st.markdown('<div id="hero"></div>', unsafe_allow_html=True)
hl, hr = st.columns([1.5, 1])
with hl:
    st.markdown('<div class="sb-hero" style="display:block;padding-top:26px"><div class="sb-eyebrow">Dinner help for busy, hungry people</div>'
                '<h1>Turn what you have into something <em>delicious.</em></h1>'
                '<p>Discover recipes, make the most of your ingredients, explore substitutions and get cooking — all grounded in a real recipe collection.</p></div>', unsafe_allow_html=True)
    with st.form("hero_form", clear_on_submit=True):
        fc = st.columns([4.2, 1.4], vertical_alignment="bottom")
        q = fc[0].text_input("What do you have — or feel like eating?", placeholder="Tell us what's in your kitchen or what you're craving...")
        go = fc[1].form_submit_button("Find my dinner  →", type="primary", use_container_width=True)
    if go and q.strip():
        ask(q.strip())
    ex = ["Find a quick chicken dinner.", "I have eggs, rice and tomatoes.", "Show me dairy-free pasta recipes.", "What can I substitute for cream?"]
    ec = st.columns(4)
    for i, t in enumerate(ex):
        if ec[i].button(t, key=f"ex{i}", use_container_width=True):
            ask(t)
with hr:
    st.markdown('<div class="sb-plate"><h4>In this collection</h4>'
                f'<div class="sb-stat"><b>{len(index):,}</b><span>recipes loaded</span></div>'
                f'<div class="sb-stat"><b>{len(index.name_index):,}</b><span>distinct ingredient names</span></div>'
                '<div class="sb-stat"><b>3</b><span>source files (title · ingredients · instructions)</span></div>'
                '<div class="note" style="margin-top:10px">No photos, nutrition or cooking times exist in this dataset, so none are shown or invented.</div></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ guided tools (4 sections)
st.markdown('<div id="tools"></div>', unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(["Discover Recipes", "Cook With My Ingredients", "Recipe Insights", "Ingredient Substitutions"])
with t1:
    with st.form("f_discover"):
        c = st.columns(2)
        dish = c[0].text_input("Dish or keywords", placeholder="penne pasta")
        inc = c[1].text_input("Must contain (comma separated)", placeholder="garlic, tomato")
        c = st.columns([2, 1])
        exc = c[0].text_input("Exclude (comma separated)", placeholder="dairy, peanuts")
        steps = c[1].number_input("Max steps (0 = any)", 0, 40, 0)
        if st.form_submit_button("Find recipes", type="primary"):
            m = f"Find {dish or 'a'} recipe" + (f" with {inc}" if inc.strip() else "") + "."
            if exc.strip():
                m += f" No {exc.replace(',', ', no ')}."
            if steps:
                m += f" At most {steps} steps."
            ask(m)
with t2:
    with st.form("f_cook"):
        have = st.text_input("Ingredients you have (comma separated, quantities optional)", placeholder="2 eggs, onions, tomatoes, rice")
        c = st.columns(3)
        exc2 = c[0].text_input("Exclude", placeholder="peanuts")
        steps2 = c[1].number_input("Max steps (0 = any)", 0, 40, 0, key="steps2")
        pref = c[2].selectbox("Preference", ["Use as many of my ingredients as possible", "Fewest additional ingredients", "Both: many of mine, few extra", "Closest recipe match", "Fewer steps"])
        if st.form_submit_button("Find recipes I can cook", type="primary") and have.strip():
            m = f"I have {have}."
            if exc2.strip():
                m += f" No {exc2.replace(',', ', no ')}."
            if steps2:
                m += f" At most {steps2} steps."
            m += {"Use as many of my ingredients as possible": " Use as many of my ingredients as possible.", "Fewest additional ingredients": " Prefer the fewest additional ingredients.",
                  "Both: many of mine, few extra": " Use as many of these as possible and prefer the fewest additional ingredients.", "Closest recipe match": " Find the closest recipe match.",
                  "Fewer steps": " Prefer fewer steps."}[pref]
            ask(m)
with t3:
    with st.form("f_insight"):
        kind = st.selectbox("Question", ["How many recipes contain…", "Most common ingredients", "Average / median number of ingredients", "Recipes with the fewest instruction steps", "Recipes with the most instruction steps"])
        c = st.columns(3)
        ings = c[0].text_input("Ingredient(s), comma separated", placeholder="garlic, onion")
        word = c[1].text_input("Dish word in title (optional)", placeholder="pasta")
        n = c[2].number_input("How many to list", 1, 20, 5)
        if st.form_submit_button("Calculate", type="primary"):
            lst = " and ".join(x.strip() for x in ings.split(",") if x.strip())
            scope = f" in {word.strip()} recipes" if word.strip() else ""
            if kind.startswith("How many") and lst:
                m = f"How many recipes contain both {lst}? What percentage is that?" if " and " in lst else f"How many recipes contain {lst}?"
            elif kind.startswith("Most common"):
                m = f"What are the {n} most common ingredients{scope}?" if not lst else f"What are the {n} most common ingredients among recipes containing {lst}?"
            elif kind.startswith("Average"):
                m = f"What is the average and median number of ingredients{scope}?"
            else:
                o = "fewest" if "fewest" in kind else "most"
                m = f"Among recipes containing {lst}, which {n} have the {o} instruction steps?" if lst else f"Which {n} {word.strip() + ' ' if word.strip() else ''}recipes have the {o} instruction steps?"
            ask(m)
with t4:
    with st.form("f_sub"):
        c = st.columns(2)
        rec = c[0].text_input("Recipe (title or ID, optional)", placeholder="banana bread  /  rb_1a2b3c4d5e6f")
        ing = c[1].text_input("Ingredient to replace or remove", placeholder="eggs")
        c = st.columns(2)
        mode = c[0].selectbox("Situation", ["I don't have it (replace)", "I can't have it (remove — hard restriction)"])
        exc3 = c[1].text_input("Other restrictions", placeholder="dairy")
        if st.form_submit_button("Suggest a substitution", type="primary") and ing.strip():
            ref = (f"recipe {rec.strip()}" if rec.strip().startswith("rb_") else f"the {rec.strip()} recipe") if rec.strip() else ""
            if mode.startswith("I can't"):
                m = f"Remove {ing.strip()}" + (f" from {ref}" if ref else "") + "."
            else:
                m = f"Replace {ing.strip()}" + (f" in {ref}" if ref else "") + "." if ref else f"What can I substitute for {ing.strip()}?"
            if exc3.strip():
                m += f" Without {exc3.strip()}."
            ask(m)

# ------------------------------------------------------------------ active context
st.markdown('<div id="chat"></div><div class="sec">Conversation</div>', unsafe_allow_html=True)
p = conv.preferences
parts = [
    '<span class="k">Available</span>' + ("".join(chip(a["name"].title(), "ok") for a in conv.available) or chip("not set", "mute")),
    '<span class="k">Excluded</span>' + ("".join(chip(x.title(), "err") for x in conv.exclusions) or chip("none", "mute")),
    '<span class="k">Max steps</span>' + chip(p["max_steps"] if p["max_steps"] is not None else "any"),
    '<span class="k">Preference</span>' + chip(GOAL_LABELS.get(p["goal"], p["goal"]) + (" + fewer steps" if p["prefer_fewer_steps"] else "")),
]
if conv.active_recipe_id:
    parts.append('<span class="k">Active recipe</span>' + chip(conv.active_recipe_id))
st.markdown('<div class="ctx">' + "".join(parts) + "</div>", unsafe_allow_html=True)
with st.expander("Edit context (remove restrictions, change ingredients or step limit)"):
    with st.form("ctx_form"):
        keep = st.multiselect("Active exclusions (untick to remove)", options=conv.exclusions, default=conv.exclusions)
        av = st.text_input("Available ingredients (comma separated)", value=", ".join(a["name"] for a in conv.available))
        c = st.columns(2)
        ms = c[0].number_input("Max steps (0 = any)", 0, 40, p["max_steps"] or 0, key="ctx_steps")
        stap = c[1].checkbox("Assume pantry staples (salt, water, pepper)", value=p["assume_staples"])
        if st.form_submit_button("Update context"):
            from core.ingredient_matcher import parse_item
            conv.exclusions = list(keep)
            if av.strip():
                conv.available = [i for i in (parse_item(x) for x in av.split(",")) if i]
            else:
                conv.available = []
            p["max_steps"] = int(ms) or None
            p["assume_staples"] = bool(stap)
            st.rerun()

# ------------------------------------------------------------------ messages
if not st.session_state.messages:
    st.markdown('<div class="banner info">Ask anything below. Follow-ups keep your context — e.g. <i>Find a penne pasta recipe</i> → <i>I cannot have dairy</i>, or <i>I have eggs, onions, tomatoes and rice… no peanuts, at most six steps</i> → <i>Exclude eggs too</i>.</div>', unsafe_allow_html=True)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"] if m["role"] == "assistant" else md_escape(m["content"]))
        render_blocks(index, m.get("blocks", []))

# ------------------------------------------------------------------ how it works
st.markdown('<div id="how"></div>', unsafe_allow_html=True)
with st.expander("How SpiceBox works (hybrid AI)"):
    st.markdown(f"""
**AI understands and reasons; Python and SQLite retrieve, calculate and verify.**
- **Retrieval & filtering:** deterministic search over {len(index):,} recipes. Hard restrictions are checked against the ingredient text *before* ranking.
- **Statistics:** computed over the complete filtered set; each answer states filters, denominator, normalisation and method. Never estimated by an AI.
- **Matching:** four 0-100 components (usage, coverage, missing, steps) × preference weights; the breakdown on each card reproduces the score.
- **Substitution:** similar recipes are retrieved; evidence is classified *Direct / Indirect / Weak or conflicting*; the optional AI reasons over only a handful of retrieved recipes and its output is validated (IDs must exist, restrictions are re-checked).
- **Optional AI:** {e(llm.status())}. Calls this session: {llm.calls} (cache hits {llm.cache_hits}, ≈{llm.chars_sent // 4:,} tokens sent). Without a key everything above still works.
- Restriction checks are text-based and are **not** an allergy or medical guarantee.
""")

q2 = st.chat_input("Tell us what's in your kitchen or what you're craving...")
if q2:
    ask(q2)
