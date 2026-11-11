"""Defensive JSON loading and runtime field detection. Never guesses silently."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

from .config import ID_KEYS, INGREDIENT_KEYS, INSTRUCTION_KEYS, TITLE_KEYS


class DataError(Exception):
    pass


class FieldMappingError(DataError):
    pass


def load_json(path) -> object:
    p = Path(path)
    if not p.exists():
        raise DataError(f"Missing file: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise DataError(f"{p.name}: not valid JSON ({e})") from e
    except (UnicodeDecodeError, OSError) as e:
        raise DataError(f"{p.name}: unreadable ({e})") from e


def iter_records(obj) -> tuple[list[tuple[object, object]], str]:
    """Return ([(source_key, record), ...], description_of_container)."""
    if isinstance(obj, list):
        return list(enumerate(obj)), "top-level list (source key = list index)"
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and sum(isinstance(v, dict) for v in vals) >= 0.5 * len(vals):
            return list(obj.items()), "top-level object keyed by record id (source key = object key)"
        lists = [k for k, v in obj.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if len(lists) == 1:
            k = lists[0]
            return [(f"{k}[{i}]", r) for i, r in enumerate(obj[k])], f"records under key '{k}' (source key = '{k}[index]')"
        raise DataError(
            "Cannot identify records in top-level object. Top-level keys (first 10): "
            f"{list(obj.keys())[:10]}. Expected a dict of record objects, a list, or exactly one key holding a list of objects."
        )
    raise DataError(f"Unsupported top-level JSON type: {type(obj).__name__}")


def _sample_dicts(records, limit=1000):
    step = max(1, len(records) // limit)
    return [r for _, r in records[::step] if isinstance(r, dict)][:limit]


def _type_name(v) -> str:
    return "null" if v is None else type(v).__name__


def describe_keys(records) -> dict:
    sample = _sample_dicts(records)
    kc, tc = Counter(), {}
    for r in sample:
        for k, v in r.items():
            kc[k] += 1
            tc.setdefault(k, Counter())[_type_name(v)] += 1
    return {"sample_size": len(sample),
            "keys": {k: {"present_in": c, "types": dict(tc[k])} for k, c in kc.most_common()}}


def _diag(records) -> str:
    d = describe_keys(records)
    lines = [f"  - {k!r}: present in {v['present_in']}/{d['sample_size']} sampled records, types {v['types']}"
             for k, v in d["keys"].items()]
    return "\n".join(lines) or "  (no object records found)"


def detect_fields(records, override: dict | None = None) -> dict:
    """Map roles -> actual keys. Raises FieldMappingError if absent or ambiguous."""
    sample = _sample_dicts(records)
    if not sample:
        raise FieldMappingError("No object-type records to inspect.")
    kc = Counter(k for r in sample for k in r)
    n = len(sample)
    mapping: dict = {}
    roles = {"title": TITLE_KEYS, "ingredients": INGREDIENT_KEYS, "instructions": INSTRUCTION_KEYS}
    for role, cands in roles.items():
        if override and override.get(role):
            key = override[role]
            if key not in kc:
                raise FieldMappingError(f"Mapping file says {role}='{key}' but that key was not found.\n{_diag(records)}")
            mapping[role] = key
            continue
        present = [k for k, c in kc.items() if k.lower() in cands and c >= 0.5 * n]
        if len(present) != 1:
            why = "no candidate field found" if not present else f"ambiguous candidates {present}"
            raise FieldMappingError(
                f"Cannot determine the '{role}' field: {why}.\nFields seen:\n{_diag(records)}\n"
                f"Create data/field_mapping.json such as "
                f'{{"title": "...", "ingredients": "...", "instructions": "..."}} and re-run.')
        mapping[role] = present[0]
    # type validation
    ok_types = {"title": {"str"}, "ingredients": {"list", "str"}, "instructions": {"list", "str"}}
    for role, key in mapping.items():
        types = Counter(_type_name(r.get(key)) for r in sample if r.get(key) is not None)
        total = sum(types.values())
        good = sum(c for t, c in types.items() if t in ok_types[role])
        if total == 0 or good < 0.8 * total:
            raise FieldMappingError(f"Field '{key}' (role {role}) has unexpected types {dict(types)}; expected {sorted(ok_types[role])}.")
    # optional genuine id
    mapping["id"] = None
    if override and override.get("id"):
        mapping["id"] = override["id"]
    else:
        for k in kc:
            if k.lower() in ID_KEYS and kc[k] >= 0.99 * n:
                vals = [r.get(k) for r in sample]
                if all(v not in (None, "") for v in vals) and len(set(map(str, vals))) == len(vals):
                    mapping["id"] = k
                    break
    return mapping
