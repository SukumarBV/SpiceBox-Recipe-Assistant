"""Central configuration (paths and candidate field names)."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = Path(os.environ.get("RECIPE_DB_PATH", PROCESSED_DIR / "recipes.db"))
INGESTION_REPORT = PROCESSED_DIR / "ingestion_report.json"
FIELD_MAPPING_FILE = ROOT / "data" / "field_mapping.json"

EXPECTED_FILES = [
    "recipes_raw_nosource_ar.json",
    "recipes_raw_nosource_epi.json",
    "recipes_raw_nosource_fn.json",
]

# Candidate field names (case-insensitive). The loader only auto-maps a role when
# EXACTLY ONE candidate is present; otherwise it stops and asks for a mapping file.
TITLE_KEYS = ("title", "name", "recipe_title", "recipe_name", "dish", "dish_name")
INGREDIENT_KEYS = ("ingredients", "ingredient", "ingredient_list", "ingredients_list")
INSTRUCTION_KEYS = ("instructions", "instruction", "directions", "direction", "steps", "method", "preparation")
ID_KEYS = ("id", "recipe_id", "_id", "uid", "uuid")

DEFAULT_TOP_N = 5
