# Architecture — SpiceBox (hybrid AI)

**AI understands and reasons. Python and SQLite retrieve, calculate and verify.**

```
raw JSON ─► data_loader (runtime field detection) ─► build_database ─► SQLite recipes.db (unchanged schema, stable IDs)
SQLite ─► RecipeIndex (in-memory: normalised names, posting lists, step counts)
UI (app.py + ui/) ─► conversation.handle_message
      rules first ─► search | analytics | ingredient_matcher | substitution      (deterministic)
      optional LLM ─► (a) interpret unclear messages → validated JSON → same deterministic code
                      (b) substitution reasoning over ≤6 retrieved recipes → validated by code
```
| Capability | Who does it |
|---|---|
| Intent / filters from natural language | Rules first; LLM (optional) only when rules are unsure (`LLM_MODE=assist`) or for every message (`always`) |
| Retrieval, restrictions, counts, rankings | Python over the full filtered set (never an LLM) |
| Matching score | Python; four components × preference weights, reproducible from the card |
| Substitution evidence | Python: neighbour recipes (Jaccard ≥ 0.25), with/without comparison, lift; evidence class computed by code |
| Substitution reasoning & revised text | LLM (optional) with ≤6 retrieved recipes; output validated (IDs ∈ retrieved set, restrictions re-checked, schema); otherwise deterministic fallback |
| Unsupported attributes (nutrition, time, price…) | Refused deterministically |

## Token economy
Never sends the dataset/DB; ≤ ~1.5k-token prompts; one call per operation; response cache; 60 s cool-down after a 429/5xx (no retries/loops); no agents, no embeddings, no vector DB; no LLM call when no evidence exists; session counter shown in the UI.

## Matching score (v2; fixes the v1 bug)
v1 used `100·(w_u·usage + w_c·coverage)`. With the "fewest missing" weights, 25 % usage + 100 % coverage + 1 step scored **77.5** and beat richer matches (the displayed numbers were consistent; the formula ignored "use as many as possible"). v2: USAGE, COVERAGE, MISSING (cap 10), STEPS components (0-100) × preference weights (balanced .50/.20/.25/.05; most-used .65/.15/.15/.05; fewest-missing .30/.20/.45/.05; both .45/.15/.35/.05; closest .35/.40/.20/.05; "fewer steps" moves 25 % weight to STEPS) plus a **relevance gate** (recipes using < min(2, n) of your ingredients rank after all that do). Score = Σ weight × displayed component, so it can always be recomputed (unit-tested).

## Evidence classes
Direct (≥3 similar recipes lacking the target use the alternative, no competing candidate) · Indirect (general-knowledge alternative; dataset only shows related use) · Weak or conflicting (few supporting recipes / competing candidates / omission only) · Insufficient.

## Windows DLL / pandas fix
The crash came from Streamlit rendering a dict/`st.dataframe`, which imports pandas (blocked by Application Control). Fix: all result tables are plain HTML (`ui/components.py`), no `st.dataframe/st.table/st.write(dict)`, no pandas import anywhere in the project, `requirements.txt` lists only Streamlit. A test runs the whole UI with `sys.modules["pandas"]=None`. (Streamlit itself still *installs* pandas as its own dependency, but the app never triggers its import.)

## Other decisions
No SQLite FTS5 (exact in-memory scan is deterministic and fast); normalisation is conservative (coconut milk ≠ milk; peanut ≈ groundnut; red onion ~ onion at 0.7 credit); step rule documented in `data_dictionary.md`.
