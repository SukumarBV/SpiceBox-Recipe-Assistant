# Data dictionary (SQLite `recipes` table)

| Column | Meaning |
|---|---|
| id | Stable ID `rb_<12 hex>`. SHA-1 of `source_file | key | canonical(title, ingredients, instructions)`; if the source has a genuine id field, of `source_file | id | <id>`. Repeatable on unchanged input. |
| title, ingredients_json, instructions_json | Original values, preserved (instructions stored exactly as in the source JSON value). |
| source_file, source_key | Origin file and dict key / list index. |
| norm_names | Newline-joined normalised ingredient names (derived). |
| ing_count | Number of non-empty, non-boilerplate ingredient lines. |
| step_count, step_method | Deterministic step rule (below). |
| content_hash | Hash of title+ingredients+instructions (duplicate reporting). |

**Step rule:** list → items; string → non-empty lines (≥2); single line → sentences; none → 0 (excluded from step rankings and from step-limit matching). Limitation: line breaks may not equal real steps.

**Normalisation:** lower-case; drop bracketed notes, quantities, units, text after first comma, size/prep words; singularise; tiny synonym list. Product-changing words are kept. Limitations: "1 can tomatoes" loses the "canned" form; free-text lines with unusual structure may normalise poorly.

Not available in the data (never invented): nutrition, cooking/prep time, servings unless present in text.
Duplicates: exact content duplicates are kept and counted in `ingestion_report.json`.
