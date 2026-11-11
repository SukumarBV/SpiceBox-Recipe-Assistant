# SpiceBox — AI Recipe Assistant

Turn what you have into something delicious. A Streamlit website that searches, analyses, matches and adapts recipes from the Recipe Box dataset (GenAI Intern Assignment). **Hybrid AI: an optional LLM understands and reasons; Python + SQLite retrieve, calculate and verify.** Works fully without any API key.

> Status: built and unit/UI-tested on a **synthetic fixture**. Dataset-dependent runs (full ingestion, real-data evaluation, deployment) are **NOT RUN** until you execute them locally. Screenshots: add yours to `docs/screenshots/` and link them here.

## Features
- **Discover Recipes** (Task 1): dish/ingredient/natural-language search, follow-ups ("I cannot have dairy"), recipe facts, counts/percentages/common ingredients/mean-median ingredient counts/step rankings with filters, denominator, normalisation and method shown. Unavailable data (nutrition, time, price) is refused, never invented.
- **Cook With My Ingredients** (Task 2): hard exclusions and step limits applied *before* ranking; reproducible 4-component score; matched / missing / unused ingredients; ineligible alternatives clearly labelled; quantities never claimed verified.
- **Ingredient Substitutions** (Task 3): evidence from similar local recipes with verified IDs, evidence class (Direct / Indirect / Weak or conflicting), full revised ingredients + instructions, predicted effects, uncertainties.
- **Recipe Insights**: the statistics tools. Conversation context (available, excluded, step limit, preference) is always visible and editable; reset any time.

## Setup (Windows PowerShell)
```
cd SpiceBox_Recipe_Assistant
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```
Put the three files from https://eightportions.com/datasets/Recipes/ into `data\raw\`: `recipes_raw_nosource_ar.json`, `recipes_raw_nosource_epi.json`, `recipes_raw_nosource_fn.json`.
```
python scripts\inspect_dataset.py
python scripts\build_database.py          # skips if data\processed\recipes.db exists; --force to rebuild
streamlit run app.py
```
**Already built the database with the previous version?** Keep `data\processed\recipes.db` — the schema and recipe IDs are unchanged; just run `streamlit run app.py`.

## Optional AI (free tier)
Copy `.env.example` to `.env` (or set environment variables / Streamlit secrets):
- Gemini via Google AI Studio: `GEMINI_API_KEY=...` (check AI Studio for the model currently free; set `LLM_MODEL`).
- Or any OpenAI-compatible endpoint: `LLM_PROVIDER=openai_compatible`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.
- `LLM_MODE=assist` (default; AI only when rules are unsure) or `always`.
The header shows AI on/off. Rate limits pause the AI for 60 s and the app continues deterministically. Keys are never stored in code. Not verified against a live key in this build (no key was available) — the client is tested with a fake transport.

## Tests and evaluation
```
python -m pytest -q                 # 38 tests on a synthetic fixture (also runs the UI with pandas blocked)
python scripts\run_evaluation.py    # REAL data: conversations E1-E8 -> evaluation\evaluation_report.md
```
Build-environment result: 38 passed (synthetic only). Real-data evaluation: NOT RUN.

## Deployment (free) — honest assessment
Local files are not visible to the cloud, and a ~266 MB database exceeds GitHub's 100 MB per-file limit. Options:
1. **Easiest, no risk:** the interviewer runs it locally with the steps above (≈5 minutes).
2. **Public site (Streamlit Community Cloud):** push the code (no data) to a public GitHub repo; host `recipes.db` (ideally gzipped — compression ratio not measured) as a **GitHub Release asset** (up to 2 GB/asset) *only if the dataset licence allows redistribution — verify it yourself; nothing here asserts a licence*. In Streamlit → App → Settings → Secrets add `DB_URL = "<direct download URL>"` (+ optional `GEMINI_API_KEY`). On first start the app downloads the DB. Risks not tested here: Community Cloud memory (~1 GB class; the app loads all recipes into RAM — synthetic 123k-recipe test used ≈140 MB but real text is larger, measure locally with Task Manager) and cold-start download time.
3. If redistribution is not allowed: do **not** upload the data; use option 1 (or a private repo, subject to current Streamlit limits).
Steps: `git init`, `git add .`, `git commit`, create a GitHub repo, `git push`; share.streamlit.io → New app → repo/branch/`app.py` → Deploy → Secrets.
`.gitignore` already excludes the raw JSON, the DB, `.env`, `secrets.toml`, venvs and caches.

## Explaining it in an interview
- **Why hybrid?** Language and culinary judgement need an LLM; counts, filters and verification must be exact, so code does them.
- **Retrieval:** title-token and ingredient-phrase search over normalised data. **Normalisation:** strip quantities/units/prep words, singularise, keep product-changing words. **Statistics:** computed on the complete filtered set, reported with denominator. **Ranking:** see `docs/architecture.md`. **Hard restrictions:** regex checks on ingredient text before ranking (not an allergy guarantee). **State:** `ConversationState` is the source of truth; the LLM only interprets. **Substitution evidence:** neighbour recipes → with/without comparison → candidates → class; AI output validated. **Unsupported claims:** refused or labelled; AI-cited IDs not in the retrieved set are discarded. **Tokens:** see architecture.

## Troubleshooting
- *DLL load failed … ccalendar / Application Control*: fixed — the app never imports pandas. If your own environment still errors, send the traceback; do not disable Windows security.
- *Database not found*: run the build command; check `data\raw` filenames.
- *Field mapping error*: create `data\field_mapping.json` as the message says.

## Limitations
Rule-based language understanding without a key; step counts depend on source line breaks; restriction checks are text-based; no photos/nutrition/times in the dataset; quantity sufficiency unverified; substitution effects are predictions; keyword (not semantic) retrieval; real-data performance and AI behaviour unverified here.

## Data attribution / AI disclosure
Dataset: Recipe Box (Ryan Lee, eightportions.com); verify its terms before redistributing. *Disclosure template:* "I used Claude to generate and iterate on the code from the assignment; I reviewed and tested it and can explain every component. [Add what you changed.]"
