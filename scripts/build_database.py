"""Ingest the three raw JSON files into SQLite (data/processed/recipes.db).

Usage:  python scripts/build_database.py [--raw-dir data/raw] [--db data/processed/recipes.db]
Raw files are only read, never modified.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config
from core.data_loader import DataError, detect_fields, iter_records, load_json
from core.database import SCHEMA
import sqlite3
from core.ingredient_normalizer import normalize_ingredient
from core.preprocessing import content_hash, instruction_steps, make_recipe_id, real_lines, to_lines


def build(raw_dir: Path, db_path: Path, override: dict | None = None, expected=None) -> dict:
    expected = expected or config.EXPECTED_FILES
    missing = [n for n in expected if not (raw_dir / n).exists()]
    if missing:
        raise DataError("Missing dataset file(s): " + ", ".join(missing) + f"\nExpected in: {raw_dir}")
    rows, skipped = [], []
    seen_ids: dict[str, str] = {}
    chash_counter: Counter = Counter()
    report = {"files": {}, "mapping": {}, "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name in expected:
        obj = load_json(raw_dir / name)
        records, desc = iter_records(obj)
        m = detect_fields(records, override)
        report["mapping"][name] = {k: v for k, v in m.items()}
        fs = {"container": desc, "records_seen": len(records), "loaded": 0, "skipped": Counter(),
              "kept_without_instructions": 0, "step_methods": Counter()}
        for key, rec in records:
            def skip(reason):
                fs["skipped"][reason] += 1
                skipped.append((name, str(key), reason))
            if not isinstance(rec, dict):
                skip("record_not_an_object")
                continue
            title = rec.get(m["title"])
            title = title.strip() if isinstance(title, str) else ""
            if not title:
                skip("missing_or_empty_title")
                continue
            lines = to_lines(rec.get(m["ingredients"]))
            if not real_lines(lines):
                skip("missing_or_empty_ingredients")
                continue
            names = [n for n in (normalize_ingredient(l) for l in lines) if n]
            if not names:
                skip("no_usable_ingredient_names")
                continue
            instr = rec.get(m["instructions"])
            steps, method = instruction_steps(instr)
            if not steps:
                fs["kept_without_instructions"] += 1
            gid = rec.get(m["id"]) if m["id"] else None
            ch = content_hash(title, lines, instr)
            rid = make_recipe_id(name, key, title, lines, instr, gid)
            n = 2
            base = rid
            while rid in seen_ids and seen_ids[rid] != ch:  # extremely unlikely collision
                rid = f"{base}-{n}"
                n += 1
            seen_ids[rid] = ch
            chash_counter[ch] += 1
            fs["step_methods"][method] += 1
            fs["loaded"] += 1
            rows.append((rid, title, name, str(key), None if gid in (None, "") else str(gid),
                         json.dumps(lines, ensure_ascii=False), json.dumps(instr, ensure_ascii=False),
                         "\n".join(dict.fromkeys(names)), len(real_lines(lines)), len(steps), method, ch))
        fs["skipped"] = dict(fs["skipped"])
        fs["step_methods"] = dict(fs["step_methods"])
        report["files"][name] = fs
        del obj, records
    dup_groups = sum(1 for c in chash_counter.values() if c > 1)
    dup_extra = sum(c - 1 for c in chash_counter.values() if c > 1)
    ids = [r[0] for r in rows]
    if len(ids) != len(set(ids)):
        raise DataError("Internal error: duplicate recipe IDs after de-duplication step.")
    report["totals"] = {
        "loaded": len(rows), "skipped": len(skipped),
        "duplicate_content_groups": dup_groups, "extra_duplicate_records_kept": dup_extra,
        "note": "Exact content duplicates are KEPT (counted once per source record) and reported here for transparency.",
    }
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(str(tmp))
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO skipped VALUES (?,?,?)", skipped)
    con.execute("INSERT INTO meta VALUES ('ingestion_report', ?)", (json.dumps(report),))
    con.execute("INSERT INTO meta VALUES ('total_recipes', ?)", (str(len(rows)),))
    con.commit()
    con.close()
    if db_path.exists():
        db_path.unlink()
    tmp.rename(db_path)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(config.RAW_DIR))
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--force", action="store_true", help="rebuild even if the database already exists")
    args = ap.parse_args()
    if Path(args.db).exists() and not args.force:
        print(f"Database already exists: {args.db}\nNothing to do (stable IDs are unchanged). Use --force to rebuild.")
        return 0
    override = json.loads(config.FIELD_MAPPING_FILE.read_text(encoding="utf-8")) if config.FIELD_MAPPING_FILE.exists() else None
    try:
        report = build(Path(args.raw_dir), Path(args.db), override)
    except DataError as e:
        print(f"ERROR: {e}")
        return 1
    Path(config.INGESTION_REPORT).parent.mkdir(parents=True, exist_ok=True)
    Path(config.INGESTION_REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    t = report["totals"]
    print(f"Built {args.db}\n  loaded: {t['loaded']}  skipped: {t['skipped']}  "
          f"duplicate content groups: {t['duplicate_content_groups']} (extra copies kept: {t['extra_duplicate_records_kept']})")
    for n, f in report["files"].items():
        print(f"  {n}: seen {f['records_seen']}, loaded {f['loaded']}, skipped {f['skipped']}, "
              f"no-instructions kept {f['kept_without_instructions']}")
    size = Path(args.db).stat().st_size / 1_048_576
    print(f"  database size: {size:.1f} MiB (needed for the deployment decision; see README)")
    print(f"  full report: {config.INGESTION_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
