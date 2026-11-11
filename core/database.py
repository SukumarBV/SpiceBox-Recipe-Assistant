"""SQLite storage + in-memory RecipeIndex used for search, analytics, matching and substitution.

The SQLite file is the persistent store. At app start the index is loaded into memory (derived
structures only: normalised names, token/name posting lists, step counts). Original ingredient and
instruction text is read back from SQLite on demand so results always show the source text.
"""
from __future__ import annotations
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from .ingredient_normalizer import normalize_ingredient, phrase_in, singularize
from .preprocessing import instruction_steps, real_lines

SCHEMA = """
CREATE TABLE recipes (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, source_file TEXT NOT NULL, source_key TEXT NOT NULL, genuine_id TEXT,
  ingredients_json TEXT NOT NULL, instructions_json TEXT, norm_names TEXT NOT NULL,
  ing_count INTEGER NOT NULL, step_count INTEGER NOT NULL, step_method TEXT, content_hash TEXT
);
CREATE TABLE skipped (source_file TEXT, source_key TEXT, reason TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""

_WORD = re.compile(r"[a-z]+")


def db_exists(path) -> bool:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        con = sqlite3.connect(str(p))
        n = con.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
        con.close()
        return n > 0
    except sqlite3.Error:
        return False


class RecipeIndex:
    def __init__(self, db_path=None):
        self.db_path = str(db_path) if db_path else None
        self.ids: list[str] = []
        self.titles: list[str] = []
        self.title_tokens: list[frozenset] = []
        self.names: list[tuple] = []
        self.raw_lower: list[str] = []
        self.steps: list[int] = []
        self.ing_counts: list[int] = []
        self.id_to_idx: dict[str, int] = {}
        self.name_index: dict[str, list[int]] = defaultdict(list)
        self.token_index: dict[str, list[int]] = defaultdict(list)
        self.token_names: dict[str, set] = defaultdict(set)
        self._full: dict[str, dict] = {}
        self.meta: dict = {}

    def __len__(self):
        return len(self.ids)

    # ------------------------------------------------------------ building
    def add(self, rid, title, ingredient_lines, instructions_value, step_count, names=None,
            source_file="", source_key="", step_method=""):
        i = len(self.ids)
        if names is None:
            names = [normalize_ingredient(l) for l in ingredient_lines]
        names = tuple(sys.intern(n) for n in dict.fromkeys(n for n in names if n))
        self.ids.append(rid)
        self.titles.append(title)
        self.title_tokens.append(frozenset(singularize(w) for w in _WORD.findall(title.lower())))
        self.names.append(names)
        self.raw_lower.append("\n".join(ingredient_lines).lower())
        self.steps.append(step_count)
        self.ing_counts.append(len(real_lines(ingredient_lines)))
        self.id_to_idx[rid] = i
        seen = set()
        for n in names:
            self.name_index[n].append(i)
            for tk in n.split():
                if tk not in seen:
                    seen.add(tk)
                    self.token_index[tk].append(i)
                self.token_names[tk].add(n)
        if self.db_path is None:  # in-memory mode keeps full records
            steps, method = instruction_steps(instructions_value)
            self._full[rid] = {"id": rid, "title": title, "source_file": source_file, "source_key": str(source_key),
                               "ingredients": list(ingredient_lines), "instructions_raw": instructions_value,
                               "instructions_steps": steps, "step_count": step_count, "step_method": step_method or method}

    @classmethod
    def from_records(cls, records):
        """records: iterable of dicts(id,title,ingredients(list),instructions,step_count?) — used by tests."""
        idx = cls(None)
        for r in records:
            steps, method = instruction_steps(r.get("instructions"))
            idx.add(r["id"], r["title"], r["ingredients"], r.get("instructions"), r.get("step_count", len(steps)),
                    source_file=r.get("source_file", "memory"), source_key=r.get("source_key", r["id"]), step_method=method)
        return idx

    @classmethod
    def from_db(cls, path):
        idx = cls(path)
        con = sqlite3.connect(str(path))
        for rid, title, ing_json, norm, sc, _ in con.execute(
                "SELECT id,title,ingredients_json,norm_names,step_count,ing_count FROM recipes ORDER BY id"):
            idx.add(rid, title, json.loads(ing_json), None, sc, names=norm.split("\n") if norm else [])
        for k, v in con.execute("SELECT key,value FROM meta"):
            idx.meta[k] = v
        con.close()
        return idx

    # ------------------------------------------------------------ access
    def full(self, rid: str) -> dict | None:
        if rid in self._full:
            return self._full[rid]
        if self.db_path is None or rid not in self.id_to_idx:
            return None
        con = sqlite3.connect(self.db_path)
        row = con.execute("SELECT id,title,source_file,source_key,ingredients_json,instructions_json,step_count,step_method "
                          "FROM recipes WHERE id=?", (rid,)).fetchone()
        con.close()
        if not row:
            return None
        raw = json.loads(row[5]) if row[5] is not None else None
        steps, _ = instruction_steps(raw)
        return {"id": row[0], "title": row[1], "source_file": row[2], "source_key": row[3],
                "ingredients": json.loads(row[4]), "instructions_raw": raw, "instructions_steps": steps,
                "step_count": row[6], "step_method": row[7]}

    # ------------------------------------------------------------ querying
    def contains(self, term: str, mode: str = "token") -> set[int]:
        """Recipes containing an ingredient. mode 'token': any normalised ingredient name contains the term
        as a whole-word phrase ('tomato' ~ 'tomato sauce'). mode 'exact': a name equals the term."""
        t = normalize_ingredient(term)
        if not t:
            return set()
        if mode == "exact":
            return set(self.name_index.get(t, ()))
        toks = t.split()
        lists = [self.token_index.get(tk, ()) for tk in toks]
        if any(len(l) == 0 for l in lists):
            return set()
        lists.sort(key=len)
        cand = set(lists[0])
        for l in lists[1:]:
            cand &= set(l)
        if len(toks) > 1:
            cand = {i for i in cand if any(phrase_in(t, n) for n in self.names[i])}
        return cand

    def violation(self, i: int, matchers) -> tuple[str, str] | None:
        text = self.raw_lower[i]
        for m in matchers:
            hit = m.search(text)
            if hit:
                return m.label, hit
        return None
