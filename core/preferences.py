"""Preference understanding + evidence-based ranking (deterministic, over the SQLite-derived index).

Gemini (or the offline lexicon below) turns a message into *structured criteria* (``Criteria``). This module
turns those criteria into database evidence: which recipes actually contain chilli / fresh vegetables / heavy
ingredients, which titles look like salads, stews, desserts, and so on.

Honesty rules
-------------
* The database has NO cuisine, meal-type, flavour, health or cooking-time columns (only title, ingredient lines
  and instruction steps). Everything here is a transparent HEURISTIC over title words and ingredient names,
  never a verified label, and results say so.
* Nothing here invents recipes, ingredients or nutrition. Terms suggested by Gemini are kept only if they
  really occur in the database.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .ingredient_normalizer import singularize

_WORD = re.compile(r"[a-z]+")
MIN_POS = 2.0          # minimum positive evidence for a recipe to count as matching a descriptive preference
CUISINE_MIN = 3.0      # one cuisine-specific title word, or ~3 signature ingredients
MEAL_MIN = 2.0
EXACT_SCAN_LIMIT = 30000


def _s(w: str) -> str:
    return singularize(w.lower())


# --------------------------------------------------------------------------- vocabulary
# hard=True  -> recipes need real positive evidence to be returned for this tag
# hard=False -> the tag only re-orders results (quick, mild)
PROFILES: dict[str, dict] = {
    "spicy": {
        "hard": True, "label": "spicy (chilli / hot-pepper evidence)",
        "pos": {"chile": 2, "chili": 1.8, "jalapeno": 2, "serrano": 2, "habanero": 2.5, "cayenne": 1.8,
                "red pepper flake": 1.8, "chili flake": 1.8, "chipotle": 2, "hot sauce": 2, "sriracha": 2,
                "harissa": 2, "sambal": 2, "hot pepper": 2, "curry paste": 1.5, "chili powder": 1.2,
                "pepper jack": 1, "tabasco": 2, "gochujang": 1.8, "kimchi": 1.2, "jerk": 2, "ancho": 1.2,
                "poblano": 0.8, "cajun": 1.2, "scotch bonnet": 2.5, "wasabi": 1, "pepperoncini": 0.8,
                "curry powder": 0.6, "garam masala": 0.6},
        "title": {"spicy": 4, "fiery": 3, "chili": 2, "chile": 2, "jerk": 3, "buffalo": 2.5, "vindaloo": 3,
                  "diavolo": 3, "diabla": 3, "arrabbiata": 3, "szechuan": 2.5, "sichuan": 2.5, "kung pao": 2.5,
                  "curry": 1.2, "cajun": 2, "harissa": 3, "sriracha": 3, "chipotle": 3, "jalapeno": 3,
                  "kimchi": 2, "habanero": 3, "sambal": 3, "gochujang": 3, "tom yum": 2, "firecracker": 2,
                  "masala": 1, "tikka": 1, "enchilada": 1},
    },
    "mild": {
        "hard": False, "label": "mild (no chilli evidence)", "neg_from": "spicy", "title": {"mild": 3},
    },
    "light": {
        "hard": True, "label": "light (fresh vegetables, lean proteins, few heavy ingredients)",
        "pos": {"lettuce": 1.2, "cucumber": 1.2, "arugula": 1, "spinach": 1, "mixed green": 1.2, "romaine": 1.2,
                "zucchini": 1, "tomato": 0.8, "radish": 0.8, "asparagus": 0.8, "green bean": 0.6, "shrimp": 1,
                "cod": 0.8, "tilapia": 0.8, "chicken breast": 0.8, "tuna": 0.6, "lemon juice": 0.6,
                "lime juice": 0.6, "avocado": 0.6, "quinoa": 1, "chickpea": 0.6, "bell pepper": 0.5,
                "cabbage": 0.6, "bok choy": 0.8, "snap pea": 0.6, "mint": 0.6, "basil": 0.5, "cilantro": 0.4,
                "yogurt": 0.5, "vegetable broth": 0.8},
        "title": {"salad": 3.5, "light": 4, "lighter": 3, "lean": 2, "grilled": 1.5, "steamed": 2.5,
                  "poached": 2, "gazpacho": 3, "ceviche": 3, "wrap": 1, "lettuce": 2, "veggie": 1.5,
                  "vegetable": 1.5, "skinny": 3, "low fat": 3, "fresh": 1, "broth": 1, "soup": 0.5,
                  "cucumber": 2, "poke": 3, "tabbouleh": 3, "summer": 2, "spring": 1, "zucchini": 1,
                  "ratatouille": 1.5},
        "neg": {"heavy cream": 1.5, "cream cheese": 1.2, "bacon": 1.2, "sour cream": 0.6, "cheddar": 0.6,
                "velveeta": 2, "shortening": 1.2, "lard": 1.2, "condensed milk": 1.5, "mayonnaise": 0.5,
                "cream of mushroom soup": 1.5, "cream of chicken soup": 1.5, "sausage": 0.8, "butter": 0.3,
                "ground beef": 0.6},
        "title_neg": {"casserole": 3, "stew": 2, "chowder": 3, "lasagna": 3, "pot pie": 3, "gratin": 3,
                      "creamy": 2.5, "cheesy": 2, "loaded": 2, "fried": 2, "baked": 0.8, "stuffed": 0.6,
                      "heavy": 2, "fondue": 3, "alfredo": 3, "carbonara": 2.5, "mac": 2, "macaroni": 1.5,
                      "burger": 1.5, "pizza": 1.5, "pie": 2.5, "cake": 4, "cookie": 4, "brownie": 4,
                      "pudding": 2.5, "fudge": 4, "bacon": 1.5, "bbq": 1, "barbecue": 1},
    },
    "refreshing": {
        "hard": True, "label": "refreshing / summery (cool, fresh, no-cook style)",
        "pos": {"cucumber": 2, "watermelon": 2, "mint": 1.5, "lettuce": 1.5, "mixed green": 1.5, "arugula": 1.2,
                "romaine": 1.5, "tomato": 0.8, "lime juice": 1, "lemon juice": 0.8, "yogurt": 0.8, "avocado": 1,
                "cantaloupe": 1.5, "honeydew": 1.5, "basil": 0.8, "cilantro": 0.6, "radish": 1, "feta": 0.6,
                "fennel": 0.8, "peach": 0.8, "mango": 0.8, "pineapple": 0.6, "jicama": 1.5, "dill": 0.6,
                "celery": 0.5},
        "title": {"salad": 3, "gazpacho": 4, "ceviche": 4, "slaw": 2.5, "chilled": 3, "cold": 2, "cucumber": 3,
                  "tabbouleh": 3, "caprese": 3, "poke": 3, "summer": 3, "refreshing": 4, "cooler": 2,
                  "lemonade": 3, "iced": 2, "watermelon": 3, "fresh": 1, "no cook": 3, "citrus": 2, "mint": 2,
                  "lime": 1, "lemon": 1, "granita": 2, "smoothie": 2, "raita": 2, "tzatziki": 2,
                  "spring roll": 2, "summer roll": 3, "wrap": 1, "bruschetta": 1.5, "panzanella": 4,
                  "kachumber": 4, "chaat": 1.5},
        "neg": {"heavy cream": 1.5, "bacon": 1, "shortening": 1},
        "title_neg": {"stew": 3, "casserole": 3, "chowder": 3, "roast": 1.5, "braised": 3, "pot pie": 3,
                      "baked": 1, "gratin": 3, "hearty": 3, "fried": 1, "fondue": 3},
    },
    "filling": {
        "hard": True, "label": "filling / hearty (substantial proteins, starches, one-pot mains)",
        "pos": {"potato": 1, "pasta": 1, "rice": 0.8, "ground beef": 1, "chuck": 1, "pork shoulder": 1,
                "sausage": 1, "bacon": 0.5, "chicken thigh": 0.8, "lentil": 1, "black bean": 1, "chickpea": 0.8,
                "kidney bean": 1, "white bean": 1, "cheddar": 0.5, "bread": 0.4, "noodle": 0.8,
                "sweet potato": 1, "lamb": 1, "beef": 0.8, "pork": 0.8, "quinoa": 0.6, "barley": 1,
                "macaroni": 1, "spaghetti": 1, "tortilla": 0.6},
        "title": {"hearty": 4, "filling": 4, "stew": 3, "casserole": 3, "chili": 2.5, "lasagna": 3, "pot pie": 3,
                  "pot roast": 3, "roast": 1.5, "loaf": 2, "burger": 2, "sandwich": 1.5, "mac": 2, "pasta": 1.5,
                  "risotto": 2.5, "curry": 1.5, "chowder": 3, "shepherd": 3, "stroganoff": 3, "bake": 1.5,
                  "baked": 1, "loaded": 2.5, "burrito": 2.5, "enchilada": 2, "meatball": 2, "dumpling": 1.5,
                  "biryani": 2.5, "paella": 2.5, "bowl": 1, "goulash": 3, "cassoulet": 3, "jambalaya": 3,
                  "gumbo": 3},
        "title_neg": {"light": 3, "dip": 2, "appetizer": 2, "snack": 2, "bite": 1.5, "smoothie": 3,
                      "cocktail": 3, "salad": 0.8},
    },
    "quick": {"hard": False, "label": "quick (fewer preparation steps; exact cooking time is not in the data)"},
    "healthy": {
        "hard": True, "label": "healthy-leaning (vegetable / whole-food evidence; not nutrition data)",
        "pos": {"kale": 1.2, "quinoa": 1, "lentil": 1, "chickpea": 1, "broccoli": 1, "spinach": 1,
                "sweet potato": 0.8, "brown rice": 1, "oat": 0.6, "salmon": 1, "tofu": 0.8, "olive oil": 0.3,
                "greek yogurt": 0.8, "avocado": 0.8, "cauliflower": 0.8, "brussels sprout": 1,
                "whole wheat": 1, "bean": 0.4, "lettuce": 0.8, "cucumber": 0.8, "zucchini": 0.8},
        "title": {"healthy": 5, "salad": 2, "veggie": 2, "vegetable": 2, "grilled": 1.5, "steamed": 2,
                  "quinoa": 2, "bowl": 1, "lean": 2, "whole grain": 2, "lentil": 1.5, "kale": 2, "light": 2,
                  "roasted": 0.8, "baked": 0.5},
        "neg": {"heavy cream": 1.2, "shortening": 1.5, "lard": 1.5, "bacon": 1.2, "velveeta": 2,
                "condensed milk": 1.5, "powdered sugar": 1.2, "brown sugar": 0.6, "sugar": 0.4,
                "corn syrup": 1.2, "margarine": 0.8, "chocolate chip": 1.5},
        "title_neg": {"cake": 4, "cookie": 4, "brownie": 4, "fried": 2.5, "creamy": 2, "cheesy": 1.5,
                      "loaded": 2, "fudge": 4, "pie": 3, "candy": 4, "bacon": 1.5, "alfredo": 3,
                      "casserole": 1, "pudding": 2.5, "frosting": 4, "milkshake": 3, "donut": 4, "pizza": 1},
    },
    "comfort": {
        "hard": True, "label": "comforting / warming (stews, soups, bakes, slow-cooked dishes)",
        "pos": {"chuck": 1, "potato": 0.6, "carrot": 0.4, "celery": 0.3, "heavy cream": 0.6, "cheddar": 0.6,
                "broth": 0.5, "stock": 0.4},
        "title": {"comfort": 4, "stew": 3, "soup": 2.5, "casserole": 3, "pot roast": 3, "chili": 2.5,
                  "chowder": 3, "pot pie": 3, "mac": 2.5, "meatloaf": 3, "lasagna": 3, "mashed": 2,
                  "creamy": 2, "cheesy": 2, "baked": 1, "roast": 2, "braised": 3, "stroganoff": 3,
                  "dumpling": 2, "cobbler": 2, "gratin": 2.5, "warm": 1.5, "shepherd": 3, "stuffing": 2},
    },
    "high_protein": {
        "hard": True, "label": "protein-rich ingredients (ingredient evidence, not nutrition data)",
        "pos": {"chicken breast": 2, "chicken": 1, "turkey": 1.2, "beef": 1, "steak": 1.5, "salmon": 1.5,
                "tuna": 1.5, "shrimp": 1.5, "egg": 0.8, "tofu": 1.5, "tempeh": 1.5, "lentil": 1.5,
                "chickpea": 1, "black bean": 1, "greek yogurt": 1.2, "cottage cheese": 1.5, "quinoa": 1,
                "edamame": 1.5, "pork tenderloin": 1.5, "cod": 1.2, "tilapia": 1.2, "ground turkey": 1.5},
        "title": {"protein": 4, "chicken": 0.8, "steak": 1.5, "salmon": 1.5, "tofu": 1.5, "lentil": 1.5,
                  "egg": 1, "shrimp": 1.5, "turkey": 1, "tuna": 1.5, "bean": 1, "omelet": 1.5,
                  "omelette": 1.5, "frittata": 1.5, "quinoa": 1},
    },
    "low_carb": {
        "hard": True, "label": "low-carb leaning (ingredient evidence, not nutrition data)",
        "pos": {"chicken": 1, "chicken breast": 1.5, "salmon": 1.5, "shrimp": 1.5, "egg": 1, "steak": 1.5,
                "beef": 1, "zucchini": 1.5, "cauliflower": 2, "broccoli": 1.2, "spinach": 1, "avocado": 1,
                "lettuce": 1, "asparagus": 1.2, "green bean": 0.8, "cabbage": 1, "mushroom": 0.8,
                "bell pepper": 0.6, "bacon": 0.8, "cheese": 0.5, "olive oil": 0.4, "almond flour": 1.5},
        "title": {"low carb": 5, "keto": 5, "zucchini": 2, "cauliflower": 2, "lettuce wrap": 3, "salad": 1.5,
                  "grilled": 1.5, "steak": 2, "omelet": 2, "frittata": 2, "skillet": 0.8},
        "neg": {"pasta": 1.5, "rice": 1.2, "potato": 1.2, "flour": 1, "sugar": 1.2, "bread": 1, "noodle": 1.5,
                "tortilla": 1, "cornmeal": 1.2, "oat": 1},
        "title_neg": {"pasta": 3, "mac": 2, "pizza": 3, "bread": 3, "cake": 4, "cookie": 4, "sandwich": 2,
                      "noodle": 3, "rice": 2, "potato": 2, "pie": 3, "muffin": 3, "pancake": 3, "waffle": 3,
                      "burrito": 2, "spaghetti": 3, "lasagna": 3},
    },
}

TAG_ALIASES = {
    "summer": "refreshing", "hot_weather": "refreshing", "cooling": "refreshing", "cool": "refreshing",
    "fresh": "refreshing", "refresh": "refreshing", "hearty": "filling", "substantial": "filling",
    "satisfying": "filling", "heavy": "filling", "fast": "quick", "easy": "quick", "simple": "quick",
    "speedy": "quick", "warming": "comfort", "cozy": "comfort", "comforting": "comfort", "fiery": "spicy",
    "hot_spicy": "spicy", "low_calorie": "light", "lighter": "light", "nutritious": "healthy",
    "protein": "high_protein", "highprotein": "high_protein", "lowcarb": "low_carb", "keto": "low_carb",
    "non_spicy": "mild", "not_spicy": "mild",
}
TAGS = tuple(PROFILES)


def normalize_tag(value) -> str | None:
    if not isinstance(value, str):
        return None
    t = re.sub(r"[\s\-]+", "_", value.strip().lower())
    t = TAG_ALIASES.get(t, t)
    return t if t in PROFILES else None


# Cuisine heuristics: title words (weight 3 each by default) and signature ingredients (weight in dict).
CUISINES: dict[str, dict] = {
    "indian": {"title": ["masala", "tikka", "tandoori", "paneer", "biryani", "dal", "dahl", "korma", "vindaloo", "samosa",
                         "naan", "chutney", "raita", "paratha", "aloo", "saag", "chana", "pakora", "curry", "butter chicken"],
               "ing": {"garam masala": 1.5, "turmeric": 0.8, "cumin": 0.5, "coriander": 0.5, "ghee": 1, "paneer": 2,
                       "curry leaf": 2, "basmati": 1, "cardamom": 0.8, "tamarind": 1, "fenugreek": 1.5,
                       "curry powder": 1, "mustard seed": 1, "cumin seed": 0.8, "chaat masala": 2}},
    "italian": {"title": ["pasta", "risotto", "marinara", "pesto", "gnocchi", "lasagna", "ravioli", "tiramisu",
                          "bruschetta", "polenta", "carbonara", "alfredo", "parmigiana", "parmesan", "spaghetti",
                          "penne", "fettuccine", "linguine", "focaccia", "caprese", "minestrone", "tortellini",
                          "italian", "ragu", "bolognese", "piccata", "cacciatore", "frittata", "panzanella"],
                "ing": {"parmesan": 0.8, "mozzarella": 1, "basil": 0.4, "oregano": 0.5, "prosciutto": 1.5,
                        "pancetta": 1.5, "ricotta": 1, "pesto": 1.5, "marinara": 1.5, "balsamic vinegar": 0.6,
                        "italian seasoning": 1.5, "pecorino": 1.5, "arborio": 2}},
    "chinese": {"title": ["stir fry", "stir fried", "lo mein", "chow mein", "wonton", "dumpling", "kung pao", "szechuan",
                          "sichuan", "chinese", "egg roll", "fried rice", "general tso", "sweet and sour", "mapo",
                          "dim sum", "hoisin"],
                "ing": {"soy sauce": 0.8, "ginger": 0.5, "sesame oil": 1, "hoisin": 2, "oyster sauce": 2,
                        "bok choy": 1.5, "rice vinegar": 1, "five spice": 2, "shaoxing": 2, "sichuan peppercorn": 2,
                        "cornstarch": 0.3, "scallion": 0.3, "green onion": 0.3}},
    "mexican": {"title": ["taco", "burrito", "enchilada", "quesadilla", "fajita", "guacamole", "salsa", "tamale",
                          "mexican", "nacho", "tostada", "carnitas", "chilaquiles", "mole", "pozole", "queso",
                          "elote", "churro", "huevos rancheros"],
                "ing": {"tortilla": 1, "jalapeno": 0.8, "cilantro": 0.5, "cumin": 0.4, "chipotle": 1.5,
                        "black bean": 0.8, "queso": 1.5, "chorizo": 1, "taco seasoning": 2, "salsa": 1,
                        "pinto bean": 1, "tomatillo": 2, "poblano": 1.5, "enchilada sauce": 2, "masa": 2,
                        "chili powder": 0.6}},
    "thai": {"title": ["thai", "pad thai", "tom yum", "tom kha", "satay", "larb", "green curry", "red curry",
                       "massaman", "panang"],
             "ing": {"fish sauce": 1.2, "coconut milk": 0.8, "lemongrass": 2, "galangal": 2.5, "thai basil": 2.5,
                     "curry paste": 1.2, "kaffir lime": 2.5, "thai chile": 1.5, "rice noodle": 0.8,
                     "palm sugar": 2, "tamarind": 0.8}},
    "japanese": {"title": ["sushi", "ramen", "udon", "tempura", "teriyaki", "miso", "japanese", "yakitori", "katsu",
                           "tonkatsu", "donburi", "onigiri", "soba", "sashimi", "okonomiyaki", "gyoza", "edamame"],
                 "ing": {"miso": 2, "mirin": 2, "sake": 1, "nori": 2, "dashi": 2.5, "wasabi": 1.5, "panko": 0.8,
                         "soy sauce": 0.4, "rice vinegar": 0.6, "sushi rice": 2.5, "bonito": 2.5}},
    "korean": {"title": ["korean", "bulgogi", "bibimbap", "kimchi", "gochujang", "japchae", "galbi", "tteokbokki"],
               "ing": {"gochujang": 2.5, "kimchi": 2, "gochugaru": 3, "sesame oil": 0.8, "sesame seed": 0.5,
                       "doenjang": 3}},
    "mediterranean": {"title": ["mediterranean", "greek", "hummus", "falafel", "tabbouleh", "tzatziki", "gyro",
                                "souvlaki", "shakshuka", "moussaka", "pita", "couscous", "spanakopita", "paella",
                                "gazpacho", "ratatouille"],
                      "ing": {"feta": 1, "olive": 0.8, "tahini": 1.2, "couscous": 1, "chickpea": 0.5, "pita": 1,
                              "oregano": 0.4, "kalamata": 1.5, "olive oil": 0.4, "lemon juice": 0.3,
                              "eggplant": 0.4, "hummus": 1.5, "tzatziki": 2}},
    "greek": {"title": ["greek", "gyro", "souvlaki", "moussaka", "tzatziki", "spanakopita", "baklava", "pastitsio"],
              "ing": {"feta": 1.5, "kalamata": 2, "tzatziki": 2, "phyllo": 1.5, "oregano": 0.5, "greek yogurt": 0.8}},
    "french": {"title": ["french", "gratin", "ratatouille", "quiche", "crepe", "bechamel", "cassoulet", "coq au vin",
                         "bouillabaisse", "souffle", "ratatouille", "tarte", "croissant", "bourguignon", "provencal",
                         "nicoise", "beurre", "vichyssoise"],
               "ing": {"dijon": 0.8, "tarragon": 1.5, "creme fraiche": 2, "brie": 1.5, "baguette": 1.5,
                       "herbes de provence": 3, "shallot": 0.6, "gruyere": 1.5, "white wine": 0.5}},
    "middle eastern": {"title": ["shawarma", "falafel", "hummus", "tabbouleh", "kebab", "kofta", "shakshuka",
                                 "baba ganoush", "fattoush", "middle eastern", "lebanese", "turkish", "moroccan",
                                 "tagine", "couscous"],
                       "ing": {"tahini": 1.5, "za'atar": 2.5, "zaatar": 2.5, "sumac": 2.5, "couscous": 1,
                               "pomegranate molasses": 3, "harissa": 1.5, "ras el hanout": 3, "chickpea": 0.4,
                               "cumin": 0.4}},
    "american": {"title": ["burger", "bbq", "barbecue", "meatloaf", "mac and cheese", "cobb", "cornbread", "pancake",
                           "coleslaw", "hot dog", "pot roast", "southern", "buffalo", "sloppy joe", "chili",
                           "american", "apple pie", "fried chicken"],
                 "ing": {"bbq sauce": 2, "cornbread": 1.5, "ranch dressing": 1.5, "hot dog": 2, "ketchup": 0.8,
                         "buttermilk": 0.5}},
    "spanish": {"title": ["spanish", "paella", "gazpacho", "tortilla espanola", "patatas bravas", "chorizo", "tapas",
                          "sangria", "churro", "pisto"],
                "ing": {"saffron": 1.5, "chorizo": 1.5, "pimenton": 3, "smoked paprika": 1.2, "manchego": 3,
                        "sherry": 1}},
    "vietnamese": {"title": ["vietnamese", "pho", "banh mi", "bun", "spring roll", "summer roll", "goi cuon"],
                   "ing": {"fish sauce": 1, "rice noodle": 1, "rice paper": 2.5, "lemongrass": 1.5, "hoisin": 0.8,
                           "mint": 0.4, "bean sprout": 1, "sriracha": 0.6}},
}
CUISINE_ALIASES = {"desi": "indian", "north indian": "indian", "south indian": "indian", "szechuan": "chinese",
                   "sichuan": "chinese", "cantonese": "chinese", "tex mex": "mexican", "tex-mex": "mexican",
                   "lebanese": "middle eastern", "turkish": "middle eastern", "moroccan": "middle eastern",
                   "arabic": "middle eastern", "persian": "middle eastern", "southern": "american",
                   "mediterranean diet": "mediterranean", "levantine": "middle eastern"}

MEALS: dict[str, dict] = {
    "breakfast": {"hard": True, "title": ["breakfast", "brunch", "pancake", "waffle", "omelet", "omelette", "frittata",
                                          "scramble", "scrambled", "oatmeal", "granola", "muffin", "french toast",
                                          "benedict", "hash", "biscuit", "crepe", "quiche", "bagel", "scone", "porridge",
                                          "overnight oats", "toast", "parfait", "shakshuka", "huevos"]},
    "snack": {"hard": True, "title": ["snack", "dip", "appetizer", "bite", "popcorn", "nacho", "crostini", "canape",
                                      "hummus", "guacamole", "salsa", "deviled", "bruschetta", "starter", "finger",
                                      "skewer", "slider", "pinwheel", "roll", "wing", "chip", "cracker", "trail mix",
                                      "party"]},
    "dessert": {"hard": True, "title": ["dessert", "cake", "cookie", "brownie", "pie", "cupcake", "pudding", "cobbler",
                                        "cheesecake", "fudge", "candy", "ice cream", "sorbet", "gelato", "mousse",
                                        "custard", "parfait", "donut", "doughnut", "tiramisu", "crumble", "trifle",
                                        "tart", "macaron", "truffle", "bark", "brittle", "popsicle", "sundae", "crisp",
                                        "blondie", "frosting", "sweet"]},
    "drink": {"hard": True, "title": ["drink", "smoothie", "cocktail", "lemonade", "punch", "tea", "coffee", "shake",
                                      "milkshake", "margarita", "sangria", "juice", "cooler", "martini", "latte",
                                      "cider", "mojito", "daiquiri", "eggnog", "mocktail", "spritzer", "lassi",
                                      "beverage", "slushie"]},
    "side": {"hard": True, "title": ["side", "mashed", "slaw", "pilaf", "coleslaw", "fries", "stuffing", "cornbread",
                                     "gratin", "biscuit", "salad", "rice", "roasted", "sauteed", "steamed", "fritter"]},
    "lunch": {"hard": False, "title": ["lunch", "sandwich", "wrap", "salad", "soup", "panini", "burger", "quesadilla",
                                       "bowl", "pita", "taco", "sub", "club", "melt", "gazpacho"],
              "excl": ["dessert", "drink", "breakfast"]},
    "dinner": {"hard": False, "title": ["dinner", "supper", "entree", "roast", "casserole", "stew", "curry", "lasagna",
                                        "steak", "chops", "meatloaf", "risotto", "stir fry", "pasta", "chicken",
                                        "salmon", "fish", "pork", "beef", "lamb", "skillet", "braised", "baked",
                                        "grilled", "tacos", "burger"],
               "excl": ["dessert", "drink", "breakfast", "snack"]},
    "meal": {"hard": False, "title": [], "excl": ["dessert", "drink", "snack"]},
}
MEAL_ALIASES = {"appetizer": "snack", "starter": "snack", "supper": "dinner", "brunch": "breakfast",
                "sweet": "dessert", "beverage": "drink", "main": "dinner", "main course": "dinner",
                "entree": "dinner", "meal": "meal", "side dish": "side", "light meal": "meal"}

_DESSERT_WORDS = {_s(w) for w in ["cake", "cookie", "brownie", "cupcake", "pudding", "cobbler", "cheesecake", "fudge",
                                  "candy", "sorbet", "gelato", "mousse", "custard", "donut", "doughnut", "tiramisu",
                                  "dessert", "frosting", "macaron", "brittle", "popsicle", "sundae", "blondie",
                                  "trifle", "crumble", "pie", "tart", "sweets"]}
_SAVORY_PIE = {_s(w) for w in ["pot", "shepherd", "meat", "chicken", "beef", "turkey", "pork", "steak", "fish", "cottage",
                               "tourtiere", "savory", "vegetable", "veggie", "spinach", "quiche", "tamale", "shepherds",
                               "lamb", "salmon", "tuna", "cheese"]}
_DRINK_WORDS = {_s(w) for w in ["smoothie", "cocktail", "lemonade", "punch", "tea", "coffee", "shake", "milkshake",
                                "margarita", "sangria", "juice", "cooler", "martini", "latte", "cider", "mojito",
                                "daiquiri", "eggnog", "mocktail", "spritzer", "lassi", "beverage", "drink", "slushie"]}
_BREAKFAST_WORDS = {_s(w) for w in ["breakfast", "brunch", "pancake", "waffle", "oatmeal", "granola", "muffin",
                                    "scone", "porridge", "bagel", "crepe"]}
_CONDIMENT_WORDS = {_s(w) for w in ["sauce", "dressing", "marinade", "rub", "seasoning", "spice", "blend", "powder", "paste",
                                    "glaze", "syrup", "jam", "jelly", "relish", "chutney", "salsa", "dip", "garam",
                                    "pickle", "vinaigrette", "mix", "condiment", "gravy"]}
_SNACK_WORDS = {_s(w) for w in ["snack", "dip", "appetizer", "popcorn", "crostini", "canape", "deviled", "nacho",
                                "starter", "pinwheel"]}


def title_kinds(ttokens) -> set[str]:
    """Rough title-based dish kind(s) used only for meal-type exclusion heuristics."""
    k = set()
    hits = ttokens & _DESSERT_WORDS
    if hits and not (hits <= {"pie", "tart"} and ttokens & _SAVORY_PIE):
        k.add("dessert")
    if ttokens & _DRINK_WORDS:
        k.add("drink")
    if ttokens & _BREAKFAST_WORDS:
        k.add("breakfast")
    if ttokens & _SNACK_WORDS:
        k.add("snack")
    return k


# --------------------------------------------------------------------------- criteria
@dataclass
class Criteria:
    tags: list = field(default_factory=list)
    cuisine: str | None = None
    meal_type: str | None = None
    evidence_terms: list = field(default_factory=list)   # ingredient names (DB-verified)
    title_keywords: list = field(default_factory=list)   # title words (DB-verified)
    avoid_terms: list = field(default_factory=list)      # soft-avoid ingredients (DB-verified)
    quick: bool = False

    def has_signals(self) -> bool:
        return bool(self.tags or self.cuisine or self.meal_type or self.evidence_terms or self.title_keywords or self.quick)

    def to_dict(self) -> dict:
        return {"tags": list(self.tags), "cuisine": self.cuisine, "meal_type": self.meal_type,
                "evidence_terms": list(self.evidence_terms), "title_keywords": list(self.title_keywords),
                "avoid_terms": list(self.avoid_terms), "quick": bool(self.quick)}

    @classmethod
    def from_dict(cls, d) -> "Criteria":
        d = d if isinstance(d, dict) else {}
        c = cls(tags=[t for t in (normalize_tag(x) for x in d.get("tags") or []) if t],
                cuisine=normalize_cuisine(d.get("cuisine")), meal_type=normalize_meal(d.get("meal_type")),
                evidence_terms=[str(x) for x in d.get("evidence_terms") or []][:8],
                title_keywords=[str(x) for x in d.get("title_keywords") or []][:8],
                avoid_terms=[str(x) for x in d.get("avoid_terms") or []][:8], quick=bool(d.get("quick")))
        c.tags = list(dict.fromkeys(c.tags))
        if c.quick and "quick" not in c.tags:
            c.tags.append("quick")
        return c

    def describe(self) -> str:
        bits = [PROFILES[t]["label"].split(" (")[0] for t in self.tags]
        if self.cuisine:
            bits.append(f"{self.cuisine} (heuristic)" if self.cuisine in CUISINES else f"{self.cuisine} (no cuisine heuristic available — not applied)")
        if self.meal_type:
            bits.append(f"{self.meal_type} (heuristic)")
        return ", ".join(bits)


def normalize_cuisine(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    v = re.sub(r"\s+", " ", value.strip().lower())
    v = CUISINE_ALIASES.get(v, v)
    return v or None


def normalize_meal(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    v = re.sub(r"\s+", " ", value.strip().lower())
    v = MEAL_ALIASES.get(v, v)
    return v if v in MEALS else None


def merge_criteria(prev: Criteria | None, new: Criteria, drop=()) -> Criteria:
    """Follow-up merge: keep earlier preferences, add/override with new ones, remove dropped tags."""
    if prev is None:
        return new
    drop_tags = {t for t in (normalize_tag(x) for x in drop) if t}
    tags = [t for t in prev.tags if t not in drop_tags]
    for t in new.tags:
        if t not in tags:
            tags.append(t)
    if "mild" in new.tags and "spicy" in tags:
        tags.remove("spicy")
    if "spicy" in new.tags and "mild" in tags:
        tags.remove("mild")
    if "light" in new.tags and "filling" in tags:
        tags.remove("filling")
    if "filling" in new.tags and "light" in tags:
        tags.remove("light")
    return Criteria(
        tags=tags,
        cuisine=new.cuisine or prev.cuisine,
        meal_type=new.meal_type or prev.meal_type,
        evidence_terms=list(dict.fromkeys(prev.evidence_terms + new.evidence_terms))[:8],
        title_keywords=list(dict.fromkeys(prev.title_keywords + new.title_keywords))[:8],
        avoid_terms=list(dict.fromkeys(prev.avoid_terms + new.avoid_terms))[:8],
        quick=("quick" in tags),
    )


# --------------------------------------------------------------------------- offline lexicon (no-LLM fallback)
_CUISINE_NOUN_SKIP = r"(?!\s+(?:toast|fries|fry|dressing|onion|yogurt|yoghurt|sausage|bread|vanilla|press|wedding|seasoning|bean|beans|cream|macaron|macarons|roast))"
_CUISINE_RE = re.compile(r"\b(indian|italian|chinese|mexican|thai|japanese|korean|mediterranean|greek|french|middle[- ]eastern|american|spanish|vietnamese|tex[- ]mex|lebanese|turkish|moroccan)\b" + _CUISINE_NOUN_SKIP, re.I)
_MEAL_RES = [("breakfast", re.compile(r"\b(breakfast|brunch)\b", re.I)), ("dessert", re.compile(r"\b(desserts?|sweet treats?)\b", re.I)),
             ("snack", re.compile(r"\b(snacks?|appetizers?|starters?|finger food)\b", re.I)), ("drink", re.compile(r"\b(drinks?|beverages?)\b", re.I)),
             ("side", re.compile(r"\bside dish(?:es)?\b", re.I)), ("lunch", re.compile(r"\blunch\b", re.I)),
             ("dinner", re.compile(r"\b(dinner|supper)\b", re.I)), ("meal", re.compile(r"\bmeals?\b", re.I))]
_LEXICON = [
    ("mild", re.compile(r"\b(not\s+(?:too\s+|very\s+)?spicy|non[- ]?spicy|mild(?:er|ly)?|less\s+spicy|kid[- ]friendly|no\s+heat)\b", re.I)),
    ("spicy", re.compile(r"\b(spicy|spiciest|spicier|fiery|piquant|hot\s+and\s+spicy|with\s+a\s+kick|some\s+heat|a\s+kick)\b", re.I)),
    ("light", re.compile(r"\b(light(?:er)?|low[- ]calorie|low[- ]cal|not\s+too\s+heavy|easy\s+on\s+the\s+stomach)\b", re.I)),
    ("refreshing", re.compile(r"\b(refreshing|cooling|summer(?:y)?|hot\s+(?:summer|weather|day|days|afternoon)|heat\s?wave|beat\s+the\s+heat|chilled|something\s+cold|cold\s+dish(?:es)?)\b", re.I)),
    ("filling", re.compile(r"\b(filling|hearty|substantial|satisfying|stick[- ]to[- ]your[- ]ribs|heavy\s+meal|big\s+meal|full\s+meal|really\s+hungry|very\s+hungry|starving)\b", re.I)),
    ("quick", re.compile(r"\b(quick(?:er|est)?|fast|speedy|in\s+a\s+hurry|short\s+on\s+time|no\s+time)\b", re.I)),
    ("healthy", re.compile(r"\b(healthy|healthier|nutritious|wholesome|clean\s+eating|nourishing)\b", re.I)),
    ("comfort", re.compile(r"\b(comfort(?:ing)?(?:\s+food)?|cozy|cosy|warming|cold\s+(?:night|day|weather)|rainy\s+day|winter)\b", re.I)),
    ("high_protein", re.compile(r"\b(high[- ]protein|protein[- ]rich|protein[- ]packed|lots\s+of\s+protein|more\s+protein)\b", re.I)),
    ("low_carb", re.compile(r"\b(low[- ]carb|keto)\b", re.I)),
]
_DIET_RE = re.compile(r"\b(vegetarian|vegan)\b", re.I)


def parse_request(text: str) -> tuple[Criteria, list[str], str]:
    """Offline lexicon: (criteria, diet exclusions, text with the descriptive phrases blanked out).

    Only used when Gemini is unavailable (or declined); never claims more than the heuristics support."""
    rest = text or ""
    c = Criteria()
    for tag, rx in _LEXICON:
        if rx.search(rest):
            if tag not in c.tags:
                c.tags.append(tag)
            rest = rx.sub(" ", rest)
    if "mild" in c.tags and "spicy" in c.tags:
        c.tags.remove("spicy")
    for meal, rx in _MEAL_RES:
        if rx.search(rest):
            if c.meal_type is None:
                c.meal_type = meal
            rest = rx.sub(" ", rest)
    m = _CUISINE_RE.search(rest)
    if m:
        c.cuisine = normalize_cuisine(m.group(1))
        rest = _CUISINE_RE.sub(" ", rest)
    diets = []
    for m in _DIET_RE.finditer(rest):
        d = m.group(1).lower()
        if d not in diets:
            diets.append(d)
    rest = _DIET_RE.sub(" ", rest)
    c.quick = "quick" in c.tags
    return c, diets, rest


# --------------------------------------------------------------------------- engine
class _Engine:
    """Per-index caches: ingredient postings, title-word postings, evidence results."""

    def __init__(self, index):
        self.index = index
        self._ing: dict[str, frozenset] = {}
        self._title: dict[str, frozenset] = {}
        self._tp: dict[str, list] | None = None
        self._ev: dict = {}
        self._cu: dict = {}
        self._me: dict = {}

    def ing(self, term: str) -> frozenset:
        s = self._ing.get(term)
        if s is None:
            s = frozenset(self.index.contains(term))
            self._ing[term] = s
        return s

    def _postings(self):
        if self._tp is None:
            tp = defaultdict(list)
            for i, toks in enumerate(self.index.title_tokens):
                for t in toks:
                    tp[t].append(i)
            self._tp = tp
        return self._tp

    def title(self, phrase: str) -> frozenset:
        s = self._title.get(phrase)
        if s is None:
            toks = [_s(w) for w in _WORD.findall(phrase.lower())]
            if not toks:
                s = frozenset()
            else:
                tp = self._postings()
                lists = sorted((tp.get(t, ()) for t in toks), key=len)
                if not lists[0]:
                    s = frozenset()
                else:
                    cand = set(lists[0])
                    for l in lists[1:]:
                        cand &= set(l)
                    s = frozenset(cand)
            self._title[phrase] = s
        return s

    # ---- evidence for descriptive tags + verified hints
    def evidence(self, c: Criteria):
        """Returns (pos, neg, per_key): aggregated maps plus one (pos, neg) pair per tag and for AI hints."""
        key = (tuple(sorted(c.tags)), tuple(c.evidence_terms), tuple(c.title_keywords), tuple(c.avoid_terms))
        hit = self._ev.get(key)
        if hit is not None:
            return hit
        per: dict[str, tuple] = {}
        for tag in c.tags:
            p = PROFILES.get(tag, {})
            tp: dict[int, float] = defaultdict(float)
            tn: dict[int, float] = defaultdict(float)
            for term, w in p.get("pos", {}).items():
                for i in self.ing(term):
                    tp[i] += w
            for word, w in p.get("title", {}).items():
                for i in self.title(word):
                    tp[i] += w
            for term, w in p.get("neg", {}).items():
                for i in self.ing(term):
                    tn[i] += w
            for word, w in p.get("title_neg", {}).items():
                for i in self.title(word):
                    tn[i] += w
            src = p.get("neg_from")
            if src:
                sp = PROFILES[src]
                for term, w in sp["pos"].items():
                    for i in self.ing(term):
                        tn[i] += w
                for word, w in sp["title"].items():
                    for i in self.title(word):
                        tn[i] += w
            per[tag] = (dict(tp), dict(tn))
        hp: dict[int, float] = defaultdict(float)
        hn: dict[int, float] = defaultdict(float)
        for term in c.evidence_terms:
            for i in self.ing(term):
                hp[i] += 1.2
        for word in c.title_keywords:
            for i in self.title(word):
                hp[i] += 2.0
        for term in c.avoid_terms:
            for i in self.ing(term):
                hn[i] += 1.5
        per["__hints__"] = (dict(hp), dict(hn))
        pos: dict[int, float] = defaultdict(float)
        neg: dict[int, float] = defaultdict(float)
        for tp, tn in per.values():
            for i, v in tp.items():
                pos[i] += v
            for i, v in tn.items():
                neg[i] += v
        out = (dict(pos), dict(neg), per)
        if len(self._ev) > 24:
            self._ev.clear()
        self._ev[key] = out
        return out

    def cuisine(self, name: str) -> dict[int, float]:
        hit = self._cu.get(name)
        if hit is not None:
            return hit
        spec = CUISINES.get(name)
        sc: dict[int, float] = defaultdict(float)
        if spec:
            for w in spec["title"]:
                for i in self.title(w):
                    sc[i] += 3.0
            for term, w in spec["ing"].items():
                for i in self.ing(term):
                    sc[i] += w
        self._cu[name] = dict(sc)
        return self._cu[name]

    def meal(self, name: str) -> dict[int, float]:
        hit = self._me.get(name)
        if hit is not None:
            return hit
        spec = MEALS.get(name)
        sc: dict[int, float] = defaultdict(float)
        if spec:
            for w in spec["title"]:
                for i in self.title(w):
                    sc[i] += 2.0
        self._me[name] = dict(sc)
        return self._me[name]


def engine(index) -> _Engine:
    e = getattr(index, "_pref_engine", None)
    if e is None:
        e = _Engine(index)
        index._pref_engine = e
    return e


def verify_hints(index, c: Criteria) -> Criteria:
    """Keep only Gemini-suggested ingredient/title hints that really occur in the database (never invented)."""
    eng = engine(index)
    c.evidence_terms = [t for t in dict.fromkeys(c.evidence_terms) if eng.ing(t)][:6]
    c.avoid_terms = [t for t in dict.fromkeys(c.avoid_terms) if eng.ing(t)][:6]
    c.title_keywords = [t for t in dict.fromkeys(c.title_keywords) if eng.title(t)][:6]
    return c


# --------------------------------------------------------------------------- retrieval + ranking
def _structural(index, i: int, c: Criteria, tags: set) -> float:
    s = 0.0
    st = index.steps[i]
    if "quick" in tags or c.quick:
        # 1-2 step records are often thin/incomplete, so the sweet spot is a short but real method (3-6 steps).
        if st <= 0:
            s -= 1.5
        elif st <= 2:
            s += 2.0
        elif st <= 6:
            s += 3.0
        else:
            s += max(0.0, 3.0 * (1 - (st - 6) / 8))
    if "filling" in tags and index.ing_counts[i] >= 8 and st >= 5:
        s += 1.0
    if ("light" in tags or "healthy" in tags) and 0 < index.ing_counts[i] <= 12:
        s += 0.4
    return s


def retrieve(index, c: Criteria, matchers=(), max_steps=None, base=None, need=5) -> dict:
    """Preference-aware retrieval.

    base=None  -> search the whole database for recipes that satisfy the descriptive criteria.
    base=set   -> candidates already chosen by dish/ingredient search; criteria filter + rank within them.
    Hard requirements are relaxed step by step (meal -> cuisine -> evidence when a base set exists) only when
    nothing matches, and every relaxation is reported in ``relaxed``.
    """
    eng = engine(index)
    n = len(index)
    tags = set(c.tags)
    pos, neg, per = eng.evidence(c)
    hard_keys = [t for t in c.tags if PROFILES[t].get("hard")]
    if c.evidence_terms or c.title_keywords:
        hard_keys.append("__hints__")
    has_ev = bool(hard_keys)
    cu = eng.cuisine(c.cuisine) if c.cuisine in CUISINES else {}
    me = eng.meal(c.meal_type) if c.meal_type in MEALS else {}
    meal_spec = MEALS.get(c.meal_type) if c.meal_type else None
    excl_kinds = set(meal_spec.get("excl", [])) if meal_spec else set()
    meal_hard = bool(meal_spec and meal_spec.get("hard"))
    if not meal_spec and (tags & {"light", "filling", "refreshing", "healthy", "comfort", "high_protein", "low_carb"}):
        excl_kinds |= {"dessert"}  # a descriptive "meal" preference never wants cakes and cookies
    unknown_cuisine = bool(c.cuisine and c.cuisine not in CUISINES)
    condiment_penalty = base is None and (c.meal_type in (None, "lunch", "dinner", "meal"))

    def ev_set(mode):
        sets = []
        for k in hard_keys:
            kp, kn = per[k]
            sets.append({i for i, v in kp.items() if v >= MIN_POS and v - kn.get(i, 0.0) > 0})
        if not sets:
            return set()
        out = set.intersection(*sets) if mode == "all" else set.union(*sets)
        return out

    # Relaxation ladder: every preference together -> drop meal heuristic -> drop cuisine heuristic ->
    # (several descriptive preferences) any one of them -> (dish/ingredient base only) re-order only.
    ev_modes = ["all"] + (["any"] if len(hard_keys) > 1 else [])
    attempts = []
    for mode in (ev_modes if has_ev else [None]):
        attempts.append((mode, bool(cu), meal_hard))
        if meal_hard:
            attempts.append((mode, bool(cu), False))
        if cu:
            attempts.append((mode, False, False))
    if has_ev and base is not None:
        attempts.append((None, False, False))
    seen_attempts, uniq = set(), []
    for a in attempts:
        if a not in seen_attempts:
            seen_attempts.add(a)
            uniq.append(a)

    relaxed: list[str] = []
    result = None
    for ai, (r_ev, r_cu, r_me) in enumerate(uniq):
        pool = set(base) if base is not None else None
        for flag, scores, thr in ((r_ev, None, None), (r_cu, cu, CUISINE_MIN), (r_me, me, MEAL_MIN)):
            if not flag:
                continue
            s = ev_set(r_ev) if scores is None else {i for i, v in scores.items() if v >= thr}
            pool = s if pool is None else pool & s
        if pool is None:
            pool = set(range(n))
        if max_steps is not None:
            pool = {i for i in pool if 0 < index.steps[i] <= max_steps}
        if excl_kinds:
            pool = {i for i in pool if not (title_kinds(index.title_tokens[i]) & excl_kinds)}
        if not pool:
            continue

        def score(i):
            v = min(pos.get(i, 0.0), 10.0) - neg.get(i, 0.0)
            v += 4.0 * min(pos.get(i, 0.0), 10.0) / max(index.ing_counts[i], 6) if pos.get(i) else 0.0
            v += 0.8 * min(cu.get(i, 0.0), 5.0) + 0.8 * min(me.get(i, 0.0), 4.0)
            if condiment_penalty and index.title_tokens[i] & _CONDIMENT_WORDS:
                v -= 3.0                   # sauces/spice mixes are rarely what "a spicy dish" or "something Indian" means
            return v + _structural(index, i, c, tags)

        scored = {i: score(i) for i in pool}
        ordered = sorted(pool, key=lambda i: (-scored[i], index.steps[i] if index.steps[i] > 0 else 10 ** 6,
                                              len(index.titles[i]), index.titles[i].lower(), index.ids[i]))
        passing, exact = [], len(ordered) <= EXACT_SCAN_LIMIT
        seen_titles, violated = set(), 0
        for i in ordered:
            if matchers and index.violation(i, matchers) is not None:
                violated += 1
                continue
            tkey = index.titles[i].strip().lower()
            if tkey in seen_titles:        # the dataset repeats some recipes; show each title once
                continue
            seen_titles.add(tkey)
            passing.append(i)
            if not exact and len(passing) >= need:
                break
        if passing:
            if has_ev and r_ev is None:
                relaxed.append("descriptive-preference evidence (no recipe in the dish/ingredient matches had enough; results are only re-ordered)")
            elif r_ev == "any" and len(hard_keys) > 1:
                relaxed.append("no recipe showed evidence for ALL your preferences together, so these match at least one (best combined score first)")
            if cu and not r_cu:
                relaxed.append(f"cuisine “{c.cuisine}” (not enough recipes with cuisine evidence; ranked only)")
            if meal_hard and not r_me:
                relaxed.append(f"meal type “{c.meal_type}”")
            result = {"order": passing, "scores": scored, "pool_size": len(pool), "exact": exact,
                      "excluded": violated if exact else None, "relaxed": relaxed}
            break
    if result is None:
        return {"order": [], "scores": {}, "pool_size": 0, "exact": True, "excluded": 0, "relaxed": relaxed}
    return result


def explain(index, i: int, c: Criteria, limit_terms: int = 4) -> list[str]:
    """Short, verifiable reasons a recipe matched (only evidence that is really in its title/ingredients)."""
    eng = engine(index)
    why: list[str] = []
    ttok = index.title_tokens[i]
    for tag in c.tags:
        p = PROFILES.get(tag, {})
        if tag == "quick":
            if index.steps[i] > 0:
                why.append(f"quick: only {index.steps[i]} instruction steps")
            continue
        ing_hits = [t for t in p.get("pos", {}) if i in eng.ing(t)]
        title_hits = [w for w in p.get("title", {}) if i in eng.title(w)]
        if ing_hits or title_hits:
            parts = []
            if title_hits:
                parts.append("title: " + ", ".join(title_hits[:2]))
            if ing_hits:
                parts.append("ingredients: " + ", ".join(ing_hits[:limit_terms]))
            why.append(f"{tag.replace('_', ' ')} — " + "; ".join(parts))
        elif tag == "mild":
            why.append("mild — no chilli/hot-pepper ingredients found")
    if c.cuisine in CUISINES:
        spec = CUISINES[c.cuisine]
        hits = [w for w in spec["title"] if i in eng.title(w)][:2] + [t for t in spec["ing"] if i in eng.ing(t)][:3]
        if hits:
            why.append(f"{c.cuisine} (heuristic) — " + ", ".join(hits))
    if c.meal_type in MEALS:
        hits = [w for w in MEALS[c.meal_type]["title"] if i in eng.title(w)][:2]
        if hits:
            why.append(f"{c.meal_type} (heuristic) — title: " + ", ".join(hits))
    extra = [t for t in c.evidence_terms if i in eng.ing(t)][:3] + [w for w in c.title_keywords if i in eng.title(w)][:2]
    if extra:
        why.append("also matches: " + ", ".join(extra))
    return why[:4]
