
"""Optional, provider-agnostic LLM client (stdlib only).
The app works fully without it.

Providers:
- gemini: Google AI Studio REST API
- openai_compatible: Groq, OpenRouter, or compatible endpoints

Keys are read from environment variables or Streamlit secrets.
Never hard-code API keys.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


class LLMError(Exception):
    pass


class LLMRateLimited(LLMError):
    pass


@dataclass
class LLMConfig:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    mode: str = "assist"


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

def load_config(secrets=None) -> LLMConfig:
    """
    Load LLM settings from environment variables or
    Streamlit secrets.

    Environment variables take priority.

    If secrets are not passed explicitly, Streamlit secrets
    are loaded automatically when available.
    """

    streamlit_secrets = secrets

    if streamlit_secrets is None:
        try:
            import streamlit as st
            streamlit_secrets = st.secrets
        except Exception:
            streamlit_secrets = None

    def get(key: str, default: str = "") -> str:
        value = os.environ.get(key)

        if value:
            return value.strip()

        try:
            if (
                streamlit_secrets is not None
                and key in streamlit_secrets
            ):
                value = str(streamlit_secrets[key]).strip()

                if value:
                    return value

        except Exception:
            pass

        return default

    gem = get("GEMINI_API_KEY")
    other = get("LLM_API_KEY")

    provider = get("LLM_PROVIDER").lower()

    if not provider:
        provider = (
            "gemini" if gem
            else "openai_compatible" if other
            else ""
        )

    mode = get("LLM_MODE", "assist").lower()

    if mode not in ("assist", "always"):
        mode = "assist"

    if provider == "gemini":
        return LLMConfig(
            provider="gemini",
            model=get("LLM_MODEL", "gemini-2.0-flash"),
            api_key=gem or other,
            base_url="",
            mode=mode
        )

    if provider in (
        "openai",
        "openai_compatible",
        "groq",
        "openrouter"
    ):
        return LLMConfig(
            provider="openai_compatible",
            model=get("LLM_MODEL"),
            api_key=other or (
                gem if provider == "openai_compatible" else ""
            ),
            base_url=get(
                "LLM_BASE_URL",
                "https://api.groq.com/openai/v1"
            ),
            mode=mode
        )

    return LLMConfig(mode=mode)


# ---------------------------------------------------------
# JSON PARSING
# ---------------------------------------------------------

def parse_json(text: str) -> dict:
    s = (text or "").strip()

    a = s.find("{")
    b = s.rfind("}")

    if a < 0 or b <= a:
        raise LLMError("Model did not return JSON.")

    try:
        data = json.loads(s[a:b + 1])

    except json.JSONDecodeError as e:
        raise LLMError(
            f"Model returned invalid JSON ({e})."
        ) from e

    if not isinstance(data, dict):
        raise LLMError("Model JSON was not an object.")

    return data


# ---------------------------------------------------------
# HTTP REQUEST
# ---------------------------------------------------------

def _http_post(url, headers, body, timeout=40):

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **headers
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            req,
            timeout=timeout
        ) as response:

            return json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as e:

        if e.code in (429, 500, 502, 503, 504):
            raise LLMRateLimited(
                f"LLM service busy or quota reached "
                f"(HTTP {e.code})."
            ) from None

        raise LLMError(
            f"LLM request failed (HTTP {e.code}). "
            "Check your key/model settings."
        ) from None

    except (
        urllib.error.URLError,
        TimeoutError,
        OSError
    ) as e:

        raise LLMError(
            f"LLM unreachable ({type(e).__name__})."
        ) from None


# ---------------------------------------------------------
# LLM CLIENT
# ---------------------------------------------------------

class LLMClient:

    COOLDOWN = 60

    def __init__(
        self,
        cfg: LLMConfig | None = None,
        transport=None
    ):

        self.cfg = cfg or LLMConfig()

        self._transport = transport or _http_post

        self._cache: dict[str, dict] = {}

        self._cool_until = 0.0

        self.calls = 0
        self.cache_hits = 0
        self.chars_sent = 0

        self.last_error = ""

    @property
    def configured(self) -> bool:

        return bool(
            self.cfg.provider
            and self.cfg.api_key
            and (
                self.cfg.provider == "gemini"
                or self.cfg.model
            )
        )

    @property
    def available(self) -> bool:

        return (
            self.configured
            and time.time() >= self._cool_until
        )

    @property
    def mode(self) -> str:
        return self.cfg.mode

    def status(self) -> str:

        if not self.configured:
            return "AI off · deterministic mode"

        if time.time() < self._cool_until:
            return (
                "AI paused (rate limit) · "
                "deterministic mode"
            )

        return f"AI on · {self.cfg.model}"

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 700
    ) -> dict:

        key = hashlib.sha1(
            (
                self.cfg.model
                + system
                + user
            ).encode("utf-8")
        ).hexdigest()

        if key in self._cache:

            self.cache_hits += 1

            return json.loads(
                json.dumps(self._cache[key])
            )

        if not self.available:
            raise LLMError(
                "LLM not configured or temporarily paused."
            )

        self.calls += 1

        self.chars_sent += len(system) + len(user)

        try:

            text = self._call(
                system,
                user,
                max_tokens
            )

            data = parse_json(text)

        except LLMRateLimited as e:

            self._cool_until = (
                time.time() + self.COOLDOWN
            )

            self.last_error = str(e)

            raise

        except LLMError as e:

            self.last_error = str(e)

            raise

        self._cache[key] = data

        return json.loads(
            json.dumps(data)
        )

    def _call(
        self,
        system,
        user,
        max_tokens
    ) -> str:

        c = self.cfg

        # GEMINI
        if c.provider == "gemini":

            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1beta/models/{c.model}:generateContent"
            )

            body = {
                "systemInstruction": {
                    "parts": [
                        {"text": system}
                    ]
                },

                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": user}
                        ]
                    }
                ],

                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json"
                }
            }

            # Gemini 2.5 models spend output tokens on hidden "thinking" by default, which can truncate the JSON.
            if "2.5" in (c.model or ""):
                body["generationConfig"]["thinkingConfig"] = {
                    "thinkingBudget": 0
                }

            data = self._transport(
                url,
                {
                    "x-goog-api-key": c.api_key
                },
                body
            )

            try:

                parts = data["candidates"][0]["content"]["parts"]

                text = "".join(
                    part.get("text", "")
                    for part in parts
                    if isinstance(part, dict)
                    and not part.get("thought")
                )

                if not text.strip():
                    raise KeyError("empty")

                return text

            except (
                KeyError,
                IndexError,
                TypeError
            ):

                raise LLMError(
                    "Unexpected response from the model."
                ) from None

        # OPENAI-COMPATIBLE PROVIDERS

        url = (
            c.base_url.rstrip("/")
            + "/chat/completions"
        )

        body = {
            "model": c.model,
            "temperature": 0.2,
            "max_tokens": max_tokens,

            "response_format": {
                "type": "json_object"
            },

            "messages": [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": user
                }
            ]
        }

        data = self._transport(
            url,
            {
                "Authorization": (
                    f"Bearer {c.api_key}"
                )
            },
            body
        )

        try:

            return (
                data["choices"][0]
                ["message"]["content"]
            )

        except (
            KeyError,
            IndexError,
            TypeError
        ):

            raise LLMError(
                "Unexpected response from the model."
            ) from None


# ---------------------------------------------------------
# GEMINI-POWERED NATURAL LANGUAGE UNDERSTANDING
# ---------------------------------------------------------

INTENTS = {
    "search",
    "match",
    "stats",
    "substitute",
    "restrictions",
    "chat"
}


from .preferences import MEALS, TAGS  # noqa: E402  (closed vocabularies shared with the ranking engine)

SYS_INTERPRET = """
You are the natural-language understanding engine for SpiceBox, a recipe assistant that searches a LOCAL database of
recipes. Each recipe has only: a title, an ingredient list and instruction steps. There are NO cuisine, meal-type,
flavour, health, nutrition, cooking-time or rating fields.

Your ONLY job: understand what the user MEANS and convert it to structured JSON search criteria.
You never answer the user, never name or invent recipes, never invent ingredients, nutrition facts, times or statistics.
A separate program retrieves and ranks real recipes from the database using your criteria.

HOW TO THINK
- Interpret the meaning, not the literal words. "I'm craving something spicy" means recipes that contain chillies,
  hot peppers or hot spices: set tags ["spicy"]; dish MUST be null (never put "spicy" in dish).
- "Light meal for a hot summer" means refreshing, light food (salads, cold soups, vegetable-forward dishes):
  tags ["light","refreshing"], meal_type "meal", dish null.
- "Filling"/"hearty"/"I'm starving" -> tag "filling". "Quick"/"easy"/"in a hurry" -> tag "quick" and quick true.
  "Healthy"/"nutritious" -> tag "healthy". "High protein" -> "high_protein". "Low carb"/"keto" -> "low_carb".
  "Comfort food"/"something warming for a cold day" -> "comfort". "Not too spicy"/"mild" -> "mild".
- "dish" is ONLY a specific named dish or food noun that should appear in a recipe title (pasta, banana bread, paratha,
  pancakes). Flavours, moods, weather, cuisines, meal types and health goals are NEVER the dish.
- Cuisine requests ("something Indian") -> cuisine "Indian", dish null.
- Dietary requirements are HARD restrictions: vegetarian, vegan, gluten-free, dairy-free, nut-free, egg-free, pork-free,
  shellfish-free, soy-free etc. go in exclude_ingredients using the group word (e.g. "vegetarian", "gluten", "dairy",
  "nut"). Individual disliked/allergic foods also go in exclude_ingredients. Only add restrictions the user states.
- Ingredients the user wants in the dish -> include_ingredients. Ingredients the user owns ("I have ...") ->
  available_ingredients with intent "match".
- evidence_terms: up to 6 ingredient names (singular, lowercase, as written in ingredient lists) whose presence is
  genuine evidence for the preference when the built-in vocabulary may not cover it (e.g. craving "tangy" ->
  ["lemon juice","vinegar","lime juice"]). title_keywords: up to 6 single recipe-title words that fit (e.g. "salad",
  "gazpacho"). avoid_terms: up to 6 ingredients that conflict with the request. Suggest only real food words; they are
  verified against the database and discarded if absent. Leave empty when the tags already cover the request.
- Follow-ups: use the conversation state. "make it vegetarian", "something lighter", "less spicy", "show me more",
  "what about with chicken" refine the previous search: set followup true and return ONLY what changed. For "more
  options"/"another" set followup true and more true. If the user abandons the old topic ("actually, I want pasta")
  set followup false. If the user cancels a preference ("not spicy anymore"), list it in drop_preferences.
- Understand informal, misspelled and transliterated English ("aloo pharata").
- Greetings and unrelated chat -> intent "chat". Questions about counts/rankings of the dataset -> "stats".
  Replacing/removing an ingredient in a recipe -> "substitute". Lifting a restriction -> remove_exclusions.
- Use null / empty arrays when something is not stated or not reasonably inferable.

Return ONLY valid JSON, no markdown.
"""


SCHEMA_HINT = """
{
  "intent": "search|match|stats|substitute|restrictions|chat",
  "cuisine": "string|null",
  "dish": "string|null",
  "include_ingredients": ["string"],
  "available_ingredients": ["string"],
  "exclude_ingredients": ["string"],
  "remove_exclusions": ["string"],
  "meal_type": "breakfast|lunch|dinner|snack|dessert|side|drink|meal|null",
  "tags": ["spicy|mild|light|refreshing|filling|quick|healthy|comfort|high_protein|low_carb"],
  "evidence_terms": ["string"],
  "title_keywords": ["string"],
  "avoid_terms": ["string"],
  "drop_preferences": ["string"],
  "followup": "boolean",
  "more": "boolean",
  "preferences": ["string"],
  "max_steps": "integer|null",
  "top_n": "integer|null",
  "preference": "most_used|fewest_missing|fewer_steps|closest|null",
  "quick": "boolean",
  "stat": {
    "type": "count|top_ingredients|avg_ingredients|rank_steps|null",
    "ingredients": ["string"],
    "order": "fewest|most"
  },
  "substitute": {
    "ingredient": "string|null",
    "recipe": "string|null"
  }
}
"""


# ---------------------------------------------------------
# INTERPRET USER MESSAGE
# ---------------------------------------------------------

_TAG_WORDS = {
    "summer", "hot_weather", "cooling", "hearty", "substantial", "fast", "easy", "simple", "warming",
    "fiery", "low_calorie", "nutritious", "protein", "keto", "fresh"
}


def interpret(
    client: LLMClient,
    text: str,
    state_summary: dict
) -> dict | None:

    """
    Interpret a user's message using the configured LLM.

    Returns a validated dictionary compatible with the
    existing conversation system, plus cuisine, meal_type
    and preferences.

    Returns None if interpretation fails.
    """

    user = (
        "Current conversation state:\n"
        + json.dumps(
            state_summary,
            ensure_ascii=False,
            separators=(",", ":")
        )
        + "\n\nUser message:\n"
        + text[:1000]
        + "\n\nReturn JSON following this schema:\n"
        + SCHEMA_HINT
    )

    try:

        d = client.complete_json(
            SYS_INTERPRET,
            user,
            max_tokens=900
        )

    except LLMError:

        return None

    # -----------------------------------------------------
    # VALIDATION HELPERS
    # -----------------------------------------------------

    def string_value(
        key,
        max_length=60
    ):

        value = d.get(key)

        if not isinstance(value, str):
            return None

        value = value.strip()

        return (
            value[:max_length]
            if value
            else None
        )

    def string_list(
        key,
        max_items=12
    ):

        value = d.get(key)

        if not isinstance(value, list):
            return []

        result = []

        for item in value[:max_items]:

            if isinstance(item, str):

                item = item.strip()[:60]

                if item and item.lower() not in {
                    x.lower() for x in result
                }:

                    result.append(item)

        return result

    def integer_value(key):

        value = d.get(key)

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 <= value <= 100
        ):

            return int(value)

        return None

    # -----------------------------------------------------
    # VALIDATE INTENT
    # -----------------------------------------------------

    intent = d.get("intent")

    if intent not in INTENTS:
        intent = "chat"

    # -----------------------------------------------------
    # VALIDATE CUISINE
    # -----------------------------------------------------

    cuisine = string_value(
        "cuisine",
        50
    )

    # -----------------------------------------------------
    # VALIDATE MEAL TYPE
    # -----------------------------------------------------

    meal_type = d.get("meal_type")

    if isinstance(meal_type, str):
        meal_type = meal_type.strip().lower()

    if meal_type not in set(MEALS):

        meal_type = None

    # -----------------------------------------------------
    # VALIDATE PREFERENCES
    # -----------------------------------------------------

    preferences = string_list(
        "preferences",
        10
    )

    # -----------------------------------------------------
    # VALIDATE STATISTICS
    # -----------------------------------------------------

    st = d.get("stat")

    if not isinstance(st, dict):
        st = {}

    stat_type = st.get("type")

    if stat_type not in {
        "count",
        "top_ingredients",
        "avg_ingredients",
        "rank_steps"
    }:

        stat_type = None

    stat_order = (
        "most"
        if st.get("order") == "most"
        else "fewest"
    )

    # -----------------------------------------------------
    # VALIDATE SUBSTITUTION
    # -----------------------------------------------------

    sub = d.get("substitute")

    if not isinstance(sub, dict):
        sub = {}

    substitute = {
        "ingredient": (
            sub["ingredient"].strip()[:60]
            if isinstance(
                sub.get("ingredient"),
                str
            )
            and sub["ingredient"].strip()
            else None
        ),

        "recipe": (
            sub["recipe"].strip()[:80]
            if isinstance(
                sub.get("recipe"),
                str
            )
            and sub["recipe"].strip()
            else None
        )
    }

    # -----------------------------------------------------
    # VALIDATE RANKING PREFERENCE
    # -----------------------------------------------------

    ranking = d.get("preference")

    if ranking not in {
        "most_used",
        "fewest_missing",
        "fewer_steps",
        "closest"
    }:

        ranking = None

    # -----------------------------------------------------
    # FINAL NORMALIZED OUTPUT
    # -----------------------------------------------------

    return {
        "intent": intent,

        # Existing conversation fields
        "dish": string_value(
            "dish",
            80
        ),

        "include": string_list(
            "include_ingredients"
        ),

        "available": string_list(
            "available_ingredients"
        ),

        "exclude": string_list(
            "exclude_ingredients"
        ),

        "remove_exclusions": string_list(
            "remove_exclusions"
        ),

        "max_steps": integer_value(
            "max_steps"
        ),

        "top_n": integer_value(
            "top_n"
        ),

        "preference": ranking,

        "quick": (
            d.get("quick")
            if isinstance(d.get("quick"), bool)
            else False
        ),

        "stat": {
            "type": stat_type,

            "ingredients": string_list(
                "ingredients"
            )[:6],

            "order": stat_order
        },

        "substitute": substitute,

        # Additional NLP fields
        "cuisine": cuisine,

        "meal_type": meal_type,

        "preferences": preferences,

        "tags": [
            tag for tag in string_list("tags", 10)
            if tag.lower().replace(" ", "_").replace("-", "_") in TAGS
            or tag.lower() in _TAG_WORDS
        ],

        "evidence_terms": string_list("evidence_terms", 6),

        "title_keywords": string_list("title_keywords", 6),

        "avoid_terms": string_list("avoid_terms", 6),

        "drop_preferences": string_list("drop_preferences", 6),

        "followup": d.get("followup") is True,

        "more": d.get("more") is True
    }
