"""Optional, provider-agnostic LLM client (stdlib only). The app works fully without it.

Providers: 'gemini' (Google AI Studio REST, free tier) or 'openai_compatible' (any /chat/completions endpoint,
e.g. Groq or OpenRouter free tiers). Keys come from environment variables or Streamlit secrets — never hard-coded.
Token economy: small prompts, response cache, one call per operation, cool-down after 429/5xx (no retry loops).
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
    mode: str = "assist"   # 'assist' = only when rules are unsure; 'always' = interpret every message


def load_config(secrets=None) -> LLMConfig:
    def get(k, default=""):
        v = os.environ.get(k)
        if v:
            return v
        try:
            if secrets is not None and k in secrets:
                return str(secrets[k])
        except Exception:
            pass
        return default

    gem = get("GEMINI_API_KEY")
    other = get("LLM_API_KEY")
    provider = get("LLM_PROVIDER").lower()
    if not provider:
        provider = "gemini" if gem else ("openai_compatible" if other else "")
    mode = get("LLM_MODE", "assist").lower()
    mode = mode if mode in ("assist", "always") else "assist"
    if provider == "gemini":
        return LLMConfig("gemini", get("LLM_MODEL", "gemini-2.0-flash"), gem or other, "", mode)
    if provider in ("openai", "openai_compatible", "groq", "openrouter"):
        return LLMConfig("openai_compatible", get("LLM_MODEL", ""), other, get("LLM_BASE_URL", "https://api.groq.com/openai/v1"), mode)
    return LLMConfig(mode=mode)


def parse_json(text: str) -> dict:
    s = (text or "").strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        raise LLMError("Model did not return JSON.")
    try:
        data = json.loads(s[a:b + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"Model returned invalid JSON ({e}).") from e
    if not isinstance(data, dict):
        raise LLMError("Model JSON was not an object.")
    return data


def _http_post(url, headers, body, timeout=40):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (429, 500, 502, 503, 504):
            raise LLMRateLimited(f"LLM service busy or quota reached (HTTP {e.code}).") from None
        raise LLMError(f"LLM request failed (HTTP {e.code}). Check your key/model settings.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"LLM unreachable ({type(e).__name__}).") from None


class LLMClient:
    COOLDOWN = 60

    def __init__(self, cfg: LLMConfig | None = None, transport=None):
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
        return bool(self.cfg.provider and self.cfg.api_key and (self.cfg.provider == "gemini" or self.cfg.model))

    @property
    def available(self) -> bool:
        return self.configured and time.time() >= self._cool_until

    @property
    def mode(self) -> str:
        return self.cfg.mode

    def status(self) -> str:
        if not self.configured:
            return "AI off · deterministic mode"
        if time.time() < self._cool_until:
            return "AI paused (rate limit) · deterministic mode"
        return f"AI on · {self.cfg.model}"

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict:
        key = hashlib.sha1((self.cfg.model + system + user).encode("utf-8")).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            return json.loads(json.dumps(self._cache[key]))
        if not self.available:
            raise LLMError("LLM not configured or temporarily paused.")
        self.calls += 1
        self.chars_sent += len(system) + len(user)
        try:
            text = self._call(system, user, max_tokens)
            data = parse_json(text)
        except LLMRateLimited as e:
            self._cool_until = time.time() + self.COOLDOWN
            self.last_error = str(e)
            raise
        except LLMError as e:
            self.last_error = str(e)
            raise
        self._cache[key] = data
        return json.loads(json.dumps(data))

    def _call(self, system, user, max_tokens) -> str:
        c = self.cfg
        if c.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{c.model}:generateContent"
            body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens, "responseMimeType": "application/json"}}
            data = self._transport(url, {"x-goog-api-key": c.api_key}, body)
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                raise LLMError("Unexpected response from the model.") from None
        url = c.base_url.rstrip("/") + "/chat/completions"
        body = {"model": c.model, "temperature": 0.2, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        data = self._transport(url, {"Authorization": f"Bearer {c.api_key}"}, body)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("Unexpected response from the model.") from None


# ---------------------------------------------------------------- understanding user messages
INTENTS = {"search", "match", "stats", "substitute", "restrictions", "chat"}
SYS_INTERPRET = ("You convert a message to a recipe app into JSON. Output ONLY a JSON object. Do not answer the question, "
                 "do not invent recipes, numbers or facts. Use null/[] when something is not stated.")
SCHEMA_HINT = ('{"intent":"search|match|stats|substitute|restrictions|chat","dish":str|null,"include_ingredients":[str],'
               '"available_ingredients":[str],"exclude_ingredients":[str],"remove_exclusions":[str],"max_steps":int|null,"top_n":int|null,'
               '"preference":"most_used|fewest_missing|fewer_steps|closest|null","quick":bool,'
               '"stat":{"type":"count|top_ingredients|avg_ingredients|rank_steps|null","ingredients":[str],"order":"fewest|most"},'
               '"substitute":{"ingredient":str|null,"recipe":str|null}}')


def interpret(client: LLMClient, text: str, state_summary: dict) -> dict | None:
    user = f"Current state: {json.dumps(state_summary, separators=(',', ':'))}\nMessage: {text[:500]}\nReturn JSON exactly like: {SCHEMA_HINT}"
    try:
        d = client.complete_json(SYS_INTERPRET, user, max_tokens=350)
    except LLMError:
        return None

    def sl(k):
        v = d.get(k)
        return [str(x)[:40] for x in v][:12] if isinstance(v, list) else []

    def ii(k):
        v = d.get(k)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 100 else None

    st = d.get("stat") if isinstance(d.get("stat"), dict) else {}
    sub = d.get("substitute") if isinstance(d.get("substitute"), dict) else {}
    intent = d.get("intent") if d.get("intent") in INTENTS else "chat"
    return {"intent": intent, "dish": (str(d["dish"])[:60] if d.get("dish") else None), "include": sl("include_ingredients"),
            "available": sl("available_ingredients"), "exclude": sl("exclude_ingredients"), "remove_exclusions": sl("remove_exclusions"),
            "max_steps": ii("max_steps"), "top_n": ii("top_n"), "preference": d.get("preference") if d.get("preference") in ("most_used", "fewest_missing", "fewer_steps", "closest") else None,
            "quick": bool(d.get("quick")),
            "stat": {"type": st.get("type") if st.get("type") in ("count", "top_ingredients", "avg_ingredients", "rank_steps") else None,
                     "ingredients": [str(x)[:40] for x in st.get("ingredients", [])][:6] if isinstance(st.get("ingredients"), list) else [],
                     "order": "most" if st.get("order") == "most" else "fewest"},
            "substitute": {"ingredient": (str(sub["ingredient"])[:40] if sub.get("ingredient") else None), "recipe": (str(sub["recipe"])[:60] if sub.get("recipe") else None)}}
