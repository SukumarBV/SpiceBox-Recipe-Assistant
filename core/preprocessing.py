"""Deterministic IDs, ingredient-line cleaning and instruction-step counting."""
from __future__ import annotations
import hashlib
import json
import re

BOILERPLATE = {"advertisement", "advertisements"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def to_lines(value) -> list[str]:
    """Convert an ingredient field (list or newline-separated string) to a list of non-empty strings.
    Original text is preserved (including boilerplate such as 'ADVERTISEMENT')."""
    if value is None:
        return []
    if isinstance(value, str):
        items = value.replace("\r\n", "\n").split("\n")
    elif isinstance(value, list):
        items = [x if isinstance(x, str) else ("" if x is None else str(x)) for x in value]
    else:
        return []
    return [i.strip() for i in items if i and i.strip()]


def real_lines(lines: list[str]) -> list[str]:
    return [l for l in lines if l.strip().lower() not in BOILERPLATE]


def instruction_steps(value) -> tuple[list[str], str]:
    """Deterministic step-splitting rule (documented in docs/data_dictionary.md).

    1. list value  -> each non-empty item is one step                      (method 'list_items')
    2. string      -> each non-empty line is one step (>=2 lines)           (method 'lines')
    3. string with a single line -> sentence split on . ! ? + capital/digit (method 'sentences')
    4. nothing usable -> 0 steps                                            (method 'missing')
    Lines that are only boilerplate ('ADVERTISEMENT') are dropped.
    Limitation: line breaks in the source may not equal real cooking steps.
    """
    if value is None:
        return [], "missing"
    if isinstance(value, list):
        items = [str(x).strip() for x in value if x is not None and str(x).strip()]
        items = [i for i in items if i.lower() not in BOILERPLATE]
        return items, ("list_items" if items else "missing")
    text = str(value).replace("\r\n", "\n")
    lines = [l.strip() for l in text.split("\n") if l.strip() and l.strip().lower() not in BOILERPLATE]
    if len(lines) >= 2:
        return lines, "lines"
    if len(lines) == 1:
        sents = [s.strip() for s in _SENT_SPLIT.split(lines[0]) if s.strip()]
        return sents, "sentences"
    return [], "missing"


def make_recipe_id(source_file: str, source_key, title, ingredients, instructions, genuine_id=None) -> str:
    """Stable ID. Uses a genuine source ID when one exists; otherwise hashes
    source filename + source key/index + stable content. Repeatable on unchanged input."""
    if genuine_id not in (None, ""):
        payload = f"{source_file}|id|{genuine_id}"
    else:
        payload = f"{source_file}|key|{source_key}|{_canon([title, ingredients, instructions])}"
    return "rb_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def content_hash(title, ingredients, instructions) -> str:
    return hashlib.sha1(_canon([title, ingredients, instructions]).encode("utf-8")).hexdigest()
