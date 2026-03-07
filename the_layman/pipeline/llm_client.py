from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from copy import deepcopy
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
        # Lazy import to avoid circular dependencies and reuse the global connection pool
        from the_layman.backend.app import STORE
        settings = STORE.get_llm_settings(user_id=user_id)
    except Exception as e:
        print(f"ERROR reading llm settings: {e}")
        import traceback
        traceback.print_exc()
        return None

    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20240620",
        "google": "gemini-2.0-flash",
    }

    def _configured_provider_order(selected: str) -> list[str]:
        ordered = [selected, "google", "openai", "anthropic", "local"]
        deduped: list[str] = []
        for p in ordered:
            if p not in deduped:
                deduped.append(p)
        return deduped

    def _provider_usable(provider: str) -> bool:
        if provider == "local":
            return bool(settings.local_model.strip())
        if provider == "openai":
            return bool(settings.openai_key.strip())
        if provider == "anthropic":
            return bool(settings.anthropic_key.strip())
        if provider == "google":
            return bool(settings.google_key.strip())
        return False

    provider = next(
        (p for p in _configured_provider_order(settings.provider) if _provider_usable(p)),
        settings.provider,
    )

    if provider == "openai":
        model = settings.openai_model.strip() or default_models["openai"]
        api_key = settings.openai_key.strip()
    elif provider == "anthropic":
        model = settings.anthropic_model.strip() or default_models["anthropic"]
        api_key = settings.anthropic_key.strip()
    elif provider == "google":
        model = settings.google_model.strip() or default_models["google"]
        if model == "gemini-1.5-pro-latest":
            # Legacy alias often returns 404 on newer Google OpenAI-compatible API versions.
            model = "gemini-2.0-flash"
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


def _google_model_candidates(model: str) -> list[str]:
    """Return ordered Google model retry candidates for OpenAI-compat endpoint."""
    candidates = [model]
    aliases = {
        "gemini-1.5-pro-latest": ["gemini-1.5-pro", "gemini-2.0-flash"],
        "gemini-1.5-pro": ["gemini-2.0-flash"],
        "gemini-pro": ["gemini-1.5-pro", "gemini-2.0-flash"],
    }
    for alt in aliases.get(model, ["gemini-2.0-flash"]):
        if alt not in candidates:
            candidates.append(alt)
    return candidates


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


def generate_json_with_debug(prompt: str, user_id: str = "default") -> tuple[dict | None, str | None]:
    cfg = get_llm_config(user_id=user_id)
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
        payloads = [payload]
        if is_openai_compat:
            payloads = []
            for candidate_model in _google_model_candidates(str(payload.get("model", ""))):
                candidate_payload = deepcopy(payload)
                candidate_payload["model"] = candidate_model
                payloads.append(candidate_payload)

        last_http_error: str | None = None
        for candidate_payload in payloads:
            try:
                raw = _make_request(f"{cfg.base_url}/chat/completions", candidate_payload, cfg.extra_headers)
                body = json.loads(raw.decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                try:
                    parsed = json.loads(content)
                    return (parsed if isinstance(parsed, dict) else None), content
                except json.JSONDecodeError:
                    return _extract_json_object(content), content
            except urllib.error.HTTPError as exc:
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = ""
                msg = f"HTTPError {exc.code}: {err_body or exc.reason}"
                # Retry only when Google-compatible endpoint rejects model name.
                if is_openai_compat and exc.code == 404 and "model" in (err_body or "").lower():
                    last_http_error = msg
                    continue
                return None, msg

        if last_http_error:
            return None, last_http_error
        return None, "Model call failed: no candidate model succeeded"

    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        return None, f"HTTPError {exc.code}: {err_body or exc.reason}"
    except urllib.error.URLError as exc:
        return None, f"URLError: {exc.reason}"
    except (TimeoutError, json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
        return None, f"Model call failed: {exc}"


def generate_json(prompt: str, user_id: str = "default") -> dict | None:
    parsed, _raw = generate_json_with_debug(prompt, user_id=user_id)
    return parsed


def model_version_tag(user_id: str = "default") -> str:
    cfg = get_llm_config(user_id=user_id)
    if not cfg:
        return "no-model-configured"
    seed = cfg.seed if cfg.seed is not None else "none"
    return f"{cfg.backend}:{cfg.model}:t={cfg.temperature}:seed={seed}"
