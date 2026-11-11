"""Inspect the three raw dataset files without assuming their schema.

Usage:  python scripts/inspect_dataset.py [--raw-dir data/raw]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config
from core.data_loader import DataError, FieldMappingError, describe_keys, detect_fields, iter_records, load_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(config.RAW_DIR))
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    override = None
    if config.FIELD_MAPPING_FILE.exists():
        override = json.loads(config.FIELD_MAPPING_FILE.read_text(encoding="utf-8"))
        print(f"Using field mapping override: {config.FIELD_MAPPING_FILE}")
    problems = 0
    for name in config.EXPECTED_FILES:
        p = raw / name
        print("=" * 78)
        print(name)
        if not p.exists():
            print(f"  MISSING: {p}\n  Place the downloaded file there and re-run.")
            problems += 1
            continue
        print(f"  size: {p.stat().st_size / 1_048_576:.1f} MiB")
        try:
            obj = load_json(p)
            print(f"  top-level JSON type: {type(obj).__name__}")
            records, desc = iter_records(obj)
        except DataError as e:
            print(f"  ERROR: {e}")
            problems += 1
            continue
        print(f"  records: {len(records)}  ({desc})")
        info = describe_keys(records)
        print(f"  sampled {info['sample_size']} object records; fields:")
        for k, v in info["keys"].items():
            print(f"    - {k!r}: present in {v['present_in']}, types {v['types']}")
        non_obj = sum(not isinstance(r, dict) for _, r in records)
        if non_obj:
            print(f"  note: {non_obj} records are not JSON objects (e.g. null) and will be skipped at build time.")
        try:
            m = detect_fields(records, override)
            print(f"  detected fields -> title: {m['title']!r}, ingredients: {m['ingredients']!r}, "
                  f"instructions: {m['instructions']!r}, genuine id field: {m['id']!r}")
            for _, r in records[:2]:
                if isinstance(r, dict):
                    t = str(r.get(m["title"]))[:60]
                    ni = len(r.get(m["ingredients"]) or []) if isinstance(r.get(m["ingredients"]), list) else "n/a"
                    print(f"  sample (compact): title={t!r} ingredients_items={ni} "
                          f"instructions_type={type(r.get(m['instructions'])).__name__}")
        except FieldMappingError as e:
            print(f"  FIELD MAPPING PROBLEM:\n{e}")
            problems += 1
    print("=" * 78)
    print("Inspection finished with problems." if problems else "Inspection OK. Next: python scripts/build_database.py")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
