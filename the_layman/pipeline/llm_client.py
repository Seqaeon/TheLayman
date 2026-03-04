from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class LLMConfig:
    backend: str          # "ollama" | "openai_compat" | "openai" | "anthropic" | "google"
    model: str
    base_url: str
    timeout_s: int
    temperature: float
    seed: int | None
    # Extra headers injected per request (e.g. Authorization, x-api-key)
    extra_headers: dict[str, str] = field(default_factory=dict)


def _get_db_config(user_id: str = "default") -> LLMConfig | None:
    """Try to load config from the DB's LlmSettings row.  Imports are lazy to
    avoid circular imports at module load time."""
    try:
        # Lazy import to avoid circular dependencies
        from the_layman.database.store import Store
        from pathlib import Path

        base = Path(__file__).resolve().parents[2]
        store = Store(base / "cache" / "the_layman.db")
        settings = store.get_llm_settings(user_id=user_id)
    except Exception:
        return None

    provider = settings.provider
    if provider == "openai":
        model = settings.openai_model.strip()
        api_key = settings.openai_key.strip()
    elif provider == "anthropic":
        model = settings.anthropic_model.strip()
        api_key = settings.anthropic_key.strip()
    elif provider == "google":
        model = settings.google_model.strip()
        api_key = settings.google_key.strip()
    else:
        model = settings.local_model.strip()
        api_key = ""

    if not model:
        return None

    timeout_s = int(os.getenv("LAYMAN_MODEL_TIMEOUT_S", "600"))
    temperature = float(os.getenv("LAYMAN_MODEL_TEMPERATURE", "0"))
    seed_raw = os.getenv("LAYMAN_MODEL_SEED", "").strip()
    seed = int(seed_raw) if seed_raw else None

    if provider == "local":
        base_url = (settings.local_base_url.rstrip("/") or
                    os.getenv("LAYMAN_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"))
        return LLMConfig(
            backend="ollama",
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            temperature=temperature,
            seed=seed,
        )

    if not api_key:
        return None

    if provider == "openai":
        return LLMConfig(
            backend="openai",
            model=model,
            base_url="https://api.openai.com/v1",
            timeout_s=timeout_s,
            temperature=temperature,
            seed=seed,
            extra_headers={"Authorization": f"Bearer {api_key}"},
        )

    if provider == "anthropic":
        return LLMConfig(
            backend="anthropic",
            model=model,
            base_url="https://api.anthropic.com/v1",
            timeout_s=timeout_s,
            temperature=temperature,
            seed=seed,
            extra_headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )

    if provider == "google":
        # Google AI Studio exposes an OpenAI-compatible endpoint
        return LLMConfig(
            backend="openai_compat",
            model=model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            timeout_s=timeout_s,
            temperature=temperature,
            seed=seed,
            extra_headers={"Authorization": f"Bearer {api_key}"},
        )

    return None


def get_llm_config(user_id: str = "default") -> LLMConfig | None:
    """Return LLMConfig, preferring DB-stored LlmSettings over env vars."""

    # 1. Try DB-stored settings first
    db_cfg = _get_db_config(user_id=user_id)
    if db_cfg:
        return db_cfg

    # 2. Fall back to legacy env-var configuration
    backend = os.getenv("LAYMAN_MODEL_BACKEND", "").strip().lower()
    model = os.getenv("LAYMAN_MODEL_NAME", "").strip()
    if not backend or not model:
        return None

    if backend == "ollama":
        base_url = os.getenv("LAYMAN_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    elif backend == "openai_compat":
        base_url = os.getenv("LAYMAN_OPENAI_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
    else:
        return None

    timeout_s = int(os.getenv("LAYMAN_MODEL_TIMEOUT_S", "600"))
    temperature = float(os.getenv("LAYMAN_MODEL_TEMPERATURE", "0"))
    seed_raw = os.getenv("LAYMAN_MODEL_SEED", "").strip()
    seed = int(seed_raw) if seed_raw else None
    return LLMConfig(
        backend=backend,
        model=model,
        base_url=base_url,
        timeout_s=timeout_s,
        temperature=temperature,
        seed=seed,
    )


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of first JSON object from noisy model text."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _make_request(url: str, payload: dict, headers: dict[str, str]) -> bytes:
    """Fire a POST request and return the raw response body."""
    all_headers = {"Content-Type": "application/json", **headers}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=all_headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _call_anthropic(cfg: LLMConfig, prompt: str) -> tuple[dict | None, str | None]:
    """Call the Anthropic Messages API directly (not OpenAI-compat)."""
    max_tokens = int(os.getenv("LAYMAN_MODEL_MAX_TOKENS", "3000"))
    payload = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "system": "Return strictly valid JSON only. Do not include any markdown or prose.",
        "temperature": cfg.temperature,
    }
    raw = _make_request(f"{cfg.base_url}/messages", payload, cfg.extra_headers)
    body = json.loads(raw.decode("utf-8"))
    content = body["content"][0]["text"]
    try:
        parsed = json.loads(content)
        return (parsed if isinstance(parsed, dict) else None), content
    except json.JSONDecodeError:
        return _extract_json_object(content), content


def generate_json_with_debug(prompt: str) -> tuple[dict | None, str | None]:
    cfg = get_llm_config()
    if not cfg:
        return None, None

    try:
        if cfg.backend == "anthropic":
            return _call_anthropic(cfg, prompt)

        if cfg.backend == "ollama":
            max_tokens = int(os.getenv("LAYMAN_MODEL_MAX_TOKENS", "3000"))
            payload = {
                "model": cfg.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "think": False,  # Disable Qwen3 chain-of-thought thinking; not needed for structured JSON
                "options": (
                    {"temperature": cfg.temperature, "num_predict": max_tokens}
                    | ({"seed": cfg.seed} if cfg.seed is not None else {})
                ),
            }
            raw = _make_request(f"{cfg.base_url}/api/generate", payload, cfg.extra_headers)
            body = json.loads(raw.decode("utf-8"))
            text = body.get("response", "")
            if not text:
                return None, ""
            try:
                parsed = json.loads(text)
                return (parsed if isinstance(parsed, dict) else None), text
            except json.JSONDecodeError:
                return _extract_json_object(text), text

        # OpenAI / openai_compat / Google AI (OpenAI-compat)
        max_tokens = int(os.getenv("LAYMAN_MODEL_MAX_TOKENS", "3000"))
        is_openai_compat = cfg.backend == "openai_compat"  # Google AI Studio doesn't support json_object mode or seed
        payload: dict = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": "Return strictly valid JSON only. Do not include any markdown or prose."},
                {"role": "user", "content": prompt},
            ],
            "temperature": cfg.temperature,
            "max_tokens": max_tokens,
        }
        if not is_openai_compat:
            # Standard OpenAI supports forced JSON mode and seed
            payload["response_format"] = {"type": "json_object"}
            if cfg.seed is not None:
                payload["seed"] = cfg.seed
        raw = _make_request(f"{cfg.base_url}/chat/completions", payload, cfg.extra_headers)
        body = json.loads(raw.decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            return (parsed if isinstance(parsed, dict) else None), content
        except json.JSONDecodeError:
            return _extract_json_object(content), content

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, ValueError):
        return None, None


def generate_json(prompt: str) -> dict | None:
    parsed, _raw = generate_json_with_debug(prompt)
    return parsed


def model_version_tag() -> str:
    cfg = get_llm_config()
    if not cfg:
        return "no-model-configured"
    seed = cfg.seed if cfg.seed is not None else "none"
    return f"{cfg.backend}:{cfg.model}:t={cfg.temperature}:seed={seed}"
