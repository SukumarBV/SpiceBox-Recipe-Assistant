"""Conservative ingredient normalisation + restriction (exclusion) matching.

Normalisation removes quantities, units, bracketed notes, text after the first comma and a
small set of preparation/size words, then singularises words. It deliberately KEEPS words that
change the product ('tomato paste', 'coconut milk', 'brown sugar', 'dried basil', 'canned').
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from functools import lru_cache

BOILERPLATE = {"advertisement", "advertisements"}

UNITS = {
    "cup", "cups", "c", "tablespoon", "tablespoons", "tbsp", "tbsps", "tbs", "tb", "teaspoon", "teaspoons", "tsp", "tsps",
    "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs", "gram", "grams", "g", "kg", "kilogram", "kilograms",
    "ml", "milliliter", "milliliters", "l", "liter", "liters", "litre", "litres", "pint", "pints", "quart", "quarts",
    "gallon", "gallons", "can", "cans", "jar", "jars", "package", "packages", "pkg", "packet", "packets", "stick", "sticks",
    "clove", "cloves", "slice", "slices", "piece", "pieces", "bunch", "bunches", "head", "heads", "bottle", "bottles", "box", "boxes",
    "fl", "fluid",
}
NOQTY_UNITS = {"pinch", "pinches", "dash", "dashes", "handful", "splash", "sprig", "sprigs"}

DROP_WORDS = {
    "chopped", "diced", "minced", "sliced", "grated", "shredded", "crushed", "melted", "softened", "peeled", "cubed",
    "beaten", "sifted", "packed", "optional", "fresh", "freshly", "large", "small", "medium", "big", "extra", "virgin",
    "boneless", "skinless", "unsalted", "salted", "finely", "coarsely", "roughly", "thinly", "thickly", "lightly",
    "mashed", "ripe", "firm", "pure", "good", "quality", "cold", "warm", "chilled", "drained", "rinsed", "trimmed",
    "halved", "quartered", "uncooked", "plain", "about", "approximately", "divided", "room", "temperature",
}
STOP_LEAD = {"of", "the", "a", "an"}
SYNONYMS = {
    "aubergine": "eggplant", "courgette": "zucchini", "garbanzo bean": "chickpea", "scallion": "green onion",
    "spring onion": "green onion", "capsicum": "bell pepper", "confectioner sugar": "powdered sugar",
    "icing sugar": "powdered sugar", "coriander leaf": "cilantro",
}
# Variant adjectives: 'red onion' counts as a (partial-credit) variant match for 'onion'.
VARIANT_ADJ = {"red", "yellow", "white", "cherry", "roma", "plum", "grape", "baby"}

IRREGULAR = {"cookies": "cookie", "pies": "pie", "brownies": "brownie", "smoothies": "smoothie", "chilies": "chili",
             "chiles": "chile", "leaves": "leaf", "calories": "calorie", "dies": "die"}
NO_SINGULAR = {"molasses", "hummus", "asparagus", "couscous", "lemongrass", "swiss", "watercress", "bass", "citrus",
               "cress", "grits", "hibiscus", "octopus", "bayleaf", "series", "species", "dairy", "pasta", "gas"}

_PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_QTY = re.compile(r"^[\d\s/.\-–½¼¾⅓⅔⅛⅜⅝⅞]+(?:\s*(?:to|or)\s*[\d\s/.½¼¾⅓⅔⅛]+)?")
_TAIL = re.compile(r"\b(?:to taste|for (?:garnish|serving|frying|dusting|greasing|drizzling|brushing)|as needed|if desired|plus more.*|divided)\b.*$")
_WORD = re.compile(r"[a-z]+")


@lru_cache(maxsize=200000)
def singularize(w: str) -> str:
    if w in IRREGULAR:
        return IRREGULAR[w]
    if w in NO_SINGULAR or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("oes"):
        return w[:-2]
    if w.endswith(("ches", "shes", "xes", "sses", "zes")):
        return w[:-2]
    if w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def split_quantity(line: str) -> tuple[str, str, str]:
    """Return (quantity_text, unit, remainder) from an ingredient line."""
    s = _PAREN.sub(" ", (line or "").lower()).replace("\u00a0", " ").strip()
    m = _QTY.match(s)
    qty = m.group(0).strip() if m else ""
    rest = s[m.end():].strip() if m else s
    unit = ""
    parts = rest.split(None, 1)
    if parts:
        w = parts[0].strip(".,")
        if (qty and w in UNITS) or w in NOQTY_UNITS:
            unit = w
            rest = parts[1] if len(parts) > 1 else ""
            parts2 = rest.split(None, 1)
            if parts2 and parts2[0].strip(".,") in UNITS and unit in ("fl", "fluid"):
                rest = parts2[1] if len(parts2) > 1 else ""
    rest = re.sub(r"^of\s+", "", rest.strip())
    return qty, unit, rest


@lru_cache(maxsize=400000)
def normalize_ingredient(line: str) -> str:
    """Canonical ingredient NAME for matching ('' if unusable)."""
    if not line:
        return ""
    low = line.lower().strip()
    if low in BOILERPLATE:
        return ""
    _, _, rest = split_quantity(low)
    rest = _TAIL.sub("", rest)
    rest = re.split(r"[,;]| - | – ", rest)[0]
    words = [singularize(w) for w in _WORD.findall(rest.replace("-", " "))]
    words = [w for w in words if w not in DROP_WORDS]
    while words and words[0] in STOP_LEAD:
        words.pop(0)
    name = " ".join(words)
    return SYNONYMS.get(name, name)


def phrase_in(term: str, name: str) -> bool:
    return f" {term} " in f" {name} "


def is_variant(avail: str, recipe_name: str) -> bool:
    a, r = set(avail.split()), set(recipe_name.split())
    return bool(a) and a < r and (r - a) <= VARIANT_ADJ


def related(t: str, n: str) -> bool:
    """Either name's words are contained in the other ('cream' ~ 'heavy cream')."""
    a, b = set(t.split()), set(n.split())
    return bool(a) and bool(b) and (a <= b or b <= a)


# ---------------------------------------------------------------- restrictions
DAIRY_SAFE = [
    "coconut milk", "coconut cream", "cream of coconut", "almond milk", "soy milk", "soymilk", "oat milk", "rice milk",
    "cashew milk", "nut milk", "peanut butter", "almond butter", "cashew butter", "sunflower seed butter", "cocoa butter",
    "shea butter", "cream of tartar", "butter bean", "butter beans", "non-dairy", "nondairy", "dairy-free", "dairy free",
    "vegan butter", "vegan cheese", "vegan cream",
]
_DAIRY = ["milk", "cream", "butter", "buttermilk", "cheese", "cheddar", "parmesan", "parmigiano", "mozzarella", "ricotta",
          "feta", "mascarpone", "gouda", "brie", "provolone", "gruyere", "romano", "pecorino", "yogurt", "yoghurt", "kefir",
          "ghee", "whey", "casein", "custard", "half and half", "ice cream", "creme fraiche", "paneer"]
_EGG = ["egg", "mayonnaise", "mayo", "meringue", "eggnog"]
_PEANUT = ["peanut", "groundnut"]
_TREE = ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia", "brazil nut", "pine nut", "chestnut",
         "praline", "marzipan"]
_GLUTEN = ["wheat", "flour", "bread", "breadcrumb", "pasta", "noodle", "spaghetti", "penne", "macaroni", "couscous", "bulgur",
           "barley", "rye", "semolina", "cracker", "soy sauce", "beer", "seitan", "lasagna", "fettuccine", "linguine", "orzo"]
_GLUTEN_SAFE = ["rice flour", "almond flour", "coconut flour", "corn flour", "cornflour", "gluten-free", "gluten free",
                "rice noodle", "rice pasta", "corn pasta", "chickpea flour", "tamari", "water chestnut"]
_SHELL = ["shrimp", "prawn", "crab", "lobster", "crayfish", "crawfish", "clam", "mussel", "oyster", "scallop", "squid", "calamari"]
_FISH = ["fish", "salmon", "tuna", "cod", "tilapia", "halibut", "trout", "anchovy", "sardine", "mackerel", "haddock", "snapper",
         "bass", "catfish", "swordfish", "fish sauce"]
_SOY = ["soy", "soya", "tofu", "tempeh", "edamame", "miso", "tamari", "shoyu", "soy sauce"]
_PORK = ["pork", "bacon", "ham", "prosciutto", "pancetta", "lard", "chorizo", "pepperoni", "salami", "sausage"]
_CUTS = ["chuck", "sirloin", "tenderloin", "ribeye", "rib eye", "short rib", "oxtail", "mutton", "goat", "rabbit", "quail",
         "pheasant", "bison", "meat", "roast beef", "pot roast"]
_MEAT = _PORK + _CUTS + ["chicken", "beef", "turkey", "lamb", "veal", "steak", "venison", "duck", "goose", "brisket", "hamburger",
                 "meatball", "hot dog", "gelatin", "gelatine"]

GROUPS: dict[str, dict] = {
    "dairy": {"terms": _DAIRY, "safe": DAIRY_SAFE},
    "egg": {"terms": _EGG, "safe": ["egg replacer", "egg substitute", "vegan egg", "vegan mayonnaise", "vegan mayo"]},
    "peanut": {"terms": _PEANUT, "safe": []},
    "tree nut": {"terms": _TREE, "safe": ["water chestnut"]},
    "nut": {"terms": _PEANUT + _TREE + ["nut"], "safe": ["water chestnut"]},
    "gluten": {"terms": _GLUTEN, "safe": _GLUTEN_SAFE},
    "shellfish": {"terms": _SHELL, "safe": []},
    "fish": {"terms": _FISH, "safe": []},
    "soy": {"terms": _SOY, "safe": []},
    "pork": {"terms": _PORK, "safe": []},
    "meat": {"terms": _MEAT, "safe": []},
    "vegetarian": {"terms": _MEAT + _FISH + _SHELL, "safe": []},
    "vegan": {"terms": _MEAT + _FISH + _SHELL + _DAIRY + _EGG + ["honey"], "safe": DAIRY_SAFE + ["vegan egg"]},
}
ALIASES = {"milk product": "dairy", "dairy product": "dairy", "lactose": "dairy", "wheat": "gluten", "tree-nut": "tree nut",
           "treenut": "tree nut", "shell fish": "shellfish"}
_DAIRY_TERMS = {"milk", "cream", "butter", "cheese", "yogurt", "buttermilk"}


def _term_pattern(term: str, plural: bool = True) -> str:
    words = [re.escape(w) for w in re.sub(r"[\s-]+", " ", term.strip().lower()).split()]
    core = r"[\s-]+".join(words)
    pat = core + (r"(?:s|es)?" if plural else "")
    if plural and term.endswith("y") and len(term) > 3:
        pat = f"(?:{pat}|{re.escape(term[:-1])}ies)"
    return pat


@dataclass(frozen=True)
class Matcher:
    label: str
    terms: tuple
    is_group: bool
    _re: re.Pattern
    _safe: re.Pattern | None

    def search(self, text: str) -> str | None:
        if self._safe is not None:
            text = self._safe.sub(" ", text)
        m = self._re.search(text)
        return m.group(0) if m else None


def canonical_exclusion(term: str) -> str:
    t = re.sub(r"[-\s]*free$", "", term.lower().strip())
    t = " ".join(singularize(w) for w in re.findall(r"[a-z]+", t))
    return ALIASES.get(t, t)


@lru_cache(maxsize=512)
def get_matcher(term: str) -> Matcher:
    t = canonical_exclusion(term)
    if t in GROUPS:
        g = GROUPS[t]
        terms, safe, is_group = g["terms"], g["safe"], True
    else:
        terms, safe, is_group = [t], (DAIRY_SAFE if t in _DAIRY_TERMS else []), False
    rx = re.compile(r"\b(?:" + "|".join(_term_pattern(x) for x in terms) + r")\b")
    sx = re.compile(r"\b(?:" + "|".join(_term_pattern(x, plural=False) for x in safe) + r")\b") if safe else None
    return Matcher(t, tuple(terms), is_group, rx, sx)


def describe_matcher(m: Matcher) -> str:
    if m.is_group:
        sample = ", ".join(m.terms[:6])
        return f"{m.label} (group: {sample}, …)"
    return m.label
