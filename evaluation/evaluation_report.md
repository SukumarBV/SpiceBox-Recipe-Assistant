# Evaluation report — STATUS: NOT RUN ON THE REAL DATASET

The real dataset was not available when this project was built, so **no real-data result exists in this file**. Nothing below is invented.

Run after building the database:

```
python scripts/run_evaluation.py
```
This overwrites this file with the actual output of conversations E1–E8 (`test_cases.json`) and PASS/FAIL invariant checks (hard exclusions never violated, step limit respected, ranking prefers more of your ingredients, displayed score reproducible, cited recipe IDs exist, unavailable attributes refused). Then add your honest "what worked / failed / why" notes.

| Case | Scenario | Status |
|---|---|---|
| E1 | Search penne pasta → exclude dairy | NOT RUN (real data) |
| E2 | Garlic AND onion count, percentage, denominator | NOT RUN |
| E3 | Eggs/onions/tomatoes/rice, no peanuts, ≤6 steps, then exclude eggs | NOT RUN |
| E4 | No-match (0-step limit) — constraints not relaxed | NOT RUN |
| E5 | Substitution with weak/conflicting evidence | NOT RUN |
| E6 | Substitution with local evidence (cream, dairy-free) | NOT RUN |
| E7 | Unsupported attribute (calories) refused | NOT RUN |
| E8 | One-shot LLM comparison | constructed example below |

Unit/UI tests run on a small **synthetic fixture** (`tests/fixtures.py`, invented for logic tests, never shipped as data). They verify code behaviour, not real-data results.

## E8 — Constructed one-shot comparison (illustration; no external model was run)
*A plain LLM with no database access* asked "How many recipes contain both garlic and onion, and what percentage of the dataset is that?" can only guess; any number it states has no denominator, no normalisation rule and no way to be checked. Asked "Which recipe has the fewest calories?" or "a high-protein recipe under 500 kcal?" it may invent nutrition values the dataset does not contain. Asked to "replace eggs in this recipe", it may cite recipes or evidence that do not exist.

*This application:* counts over every loaded recipe and reports denominator, normalisation, exact-name comparison and example IDs; refuses nutrition/time questions ("the dataset has no nutrition data … I won't estimate it"); and for substitutions only cites recipe IDs retrieved from the database (IDs an AI cites that were not retrieved are discarded) and classifies evidence as Direct / Indirect / Weak or conflicting.
