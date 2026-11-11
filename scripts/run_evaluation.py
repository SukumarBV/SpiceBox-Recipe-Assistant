"""Run evaluation conversations against the REAL local database and write evaluation/evaluation_report.md.
Usage: python scripts/run_evaluation.py"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config
from core.conversation import ConversationState, get_matcher, handle_message
from core.database import RecipeIndex, db_exists

ROOT = Path(__file__).resolve().parent.parent
CONSTRUCTED = """## Constructed example: grounded system vs. ungrounded one-shot answer
*Clearly labelled illustration, not a recorded model output.* Ask a plain LLM (no database access): "How many recipes contain both garlic and onion, and what percentage of the dataset is that?" It cannot read the 3 JSON files, so any number it states would be a guess presented as fact, with no stated denominator or normalisation. This application instead computes the count over every loaded recipe, reports the denominator, the normalisation rule (whole-word phrase match on normalised ingredient names), an exact-name comparison and example recipe IDs (see E2). If the data cannot answer (e.g. nutrition or cooking time), it says so instead of inventing a value.
"""


def check(case, turns_out, st, index):
    res = {}
    last = turns_out[-1]["blocks"]
    for c in case["checks"]:
        try:
            if c == "turn2_keeps_search_filters":
                res[c] = bool(st.filters["dish_tokens"])
            elif c == "no_result_violates_exclusions":
                ms = [get_matcher(e) for e in st.exclusions]
                ids = [i["id"] for i in last[0]["items"]] if last and last[0]["type"] == "recipes" else [r["id"] for r in last[0]["result"]["eligible"]]
                res[c] = all(index.violation(index.id_to_idx[i], ms) is None for i in ids)
            elif c == "max_steps_respected":
                lim = st.preferences["max_steps"]
                res[c] = all(0 < r["steps"] <= lim for r in last[0]["result"]["eligible"])
            elif c == "stats_reports_denominator_and_normalization":
                r = last[0]["result"]
                res[c] = r["denominator"] > 0 and bool(r["normalization"])
            elif c == "rank_rows_sorted_ascending":
                s = [r["steps"] for r in last[0]["result"]["rows"]]
                res[c] = s == sorted(s)
            elif c == "rows_have_ids":
                res[c] = all(r["id"].startswith("rb_") for r in last[0]["result"]["rows"])
            elif c == "uncertainty_labelled":
                r = last[0]["result"] if last else None
                res[c] = (not last) or (r.get("overall_evidence") not in ("strong",) or any(s["notes"] for s in r["slots"]))
            elif c == "ranking_prefers_more_ingredients":
                el = last[0]["result"]["eligible"] if last else []
                first_turn = turns_out[0]["blocks"][0]["result"]["eligible"]
                best = max((r["n_used"] for r in first_turn), default=0)
                res[c] = bool(first_turn) and first_turn[0]["n_used"] == best and [r["partial_only"] for r in first_turn] == sorted(r["partial_only"] for r in first_turn)
            elif c == "score_reproducible":
                from core.ingredient_matcher import combine, weights_for
                p_ = st.preferences
                w = weights_for(p_["goal"], p_["prefer_fewer_steps"])
                res[c] = all(r["score"] == combine(r["components"], w) for r in last[0]["result"]["eligible"])
            elif c == "cited_ids_exist":
                if not last:
                    res[c] = True   # nothing was proposed (recipe not found / refused): no IDs to verify
                    continue
                r = last[0]["result"]
                ids = [i for s_ in r["slots"] for grp in (s_["candidates"], s_["hints"]) for x in grp for i in x["cited_ids"]] + r.get("llm", {}).get("cited_ids", [])
                res[c] = all(i in index.id_to_idx for i in ids)
            elif c == "no_result_violates_exclusions_in_revision":
                r = last[0]["result"]
                res[c] = (not r["revision_supported"]) or not r["remaining_violations"]
            elif c == "refuses_unavailable":
                res[c] = (not last) and "no nutrition data" in turns_out[-1]["text"].lower()
            elif c == "no_eligible_and_alternatives_labelled":
                r = last[0]["result"]
                res[c] = r["n_eligible"] == 0 and all(a["violations"] for a in r["alternatives"])
        except Exception as e:  # record as failure with reason
            res[c] = f"ERROR {type(e).__name__}: {e}"
    return res


def summarize(blocks):
    out = []
    for b in blocks:
        t = b["type"]
        if t == "recipes":
            out += [f"- `{i['id']}` {i['title']} ({i['steps']} steps)" for i in b["items"]]
        elif t == "stats":
            r = b["result"]
            out.append(f"- kind={r['kind']} filters={r['filters']} analysed={r['n_analyzed']} denominator={r['denominator']}")
            if r["kind"] == "count":
                out.append(f"- count={r['count']} percentage={r['percentage']}% exact-name count={r['exact_name_count']}")
            for row in r.get("rows", []):
                out.append(f"  - {row}")
        elif t == "matches":
            r = b["result"]
            out.append(f"- eligible={r['n_eligible']} candidates={r['n_candidates']} excluded_by_steps={r['excluded_by_steps']} excluded_by_exclusion={r['excluded_by_exclusion']}")
            out += [f"  - `{x['id']}` {x['title']} score={x['score']} missing={len(x['missing'])} steps={x['steps']}" for x in r["eligible"]]
            out += [f"  - INELIGIBLE `{x['id']}` {x['title']}: {x['violations']}" for x in r["alternatives"]]
        elif t == "substitution":
            r = b["result"]
            out.append(f"- overall evidence={r['overall_evidence']} revision_supported={r['revision_supported']} remaining_violations={len(r['remaining_violations'])}")
            for s in r["slots"]:
                out.append(f"  - target {s['target']}: neighbours={s['neighbors']} strength={s['strength']} competing={s['competing']} chosen={(s['chosen'] or {}).get('name', (s['chosen'] or {}).get('kind'))} cited={[c for c in (s['chosen'] or {}).get('cited_ids', [])]}")
    return "\n".join(out)


def main():
    if not db_exists(config.DB_PATH):
        print("Database not found. Run scripts/build_database.py first.")
        return 1
    index = RecipeIndex.from_db(config.DB_PATH)
    cases = json.loads((ROOT / "evaluation" / "test_cases.json").read_text(encoding="utf-8"))["cases"]
    lines = [f"# Evaluation report\n\nGenerated {time.strftime('%Y-%m-%d %H:%M:%S')} against `{config.DB_PATH}` ({len(index):,} recipes). Values below are ACTUAL outputs; invariant checks are PASS/FAIL.\n"]
    for case in cases:
        st = ConversationState()
        outs = []
        lines.append(f"## {case['id']} — {case['title']} (Task {case['task']})\n")
        for t in case["turns"]:
            r = handle_message(t, st, index)
            outs.append({"text": r.text, "blocks": r.blocks})
            lines.append(f"**User:** {t}\n\n**Assistant:** {r.text}\n\n{summarize(r.blocks)}\n")
        chk = check(case, outs, st, index)
        lines.append("**Checks:** " + ", ".join(f"{k}: {'PASS' if v is True else ('FAIL' if v is False else v)}" for k, v in chk.items()) + "\n")
        lines.append("**What worked / failed / why:** _fill in after reading the output above (be honest about wrong or odd results)._\n")
    lines.append(CONSTRUCTED)
    (ROOT / "evaluation" / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote evaluation/evaluation_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
