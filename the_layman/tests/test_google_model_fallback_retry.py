from __future__ import annotations

import io
import json
import urllib.error

from the_layman.pipeline import llm_client
from the_layman.pipeline.llm_client import LLMConfig, generate_json_with_debug


def _http_404_model_not_found() -> urllib.error.HTTPError:
    body = {
        "error": {
            "code": 404,
            "message": "models/gemini-1.5-pro-latest is not found",
            "status": "NOT_FOUND",
        }
    }
    return urllib.error.HTTPError(
        url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        code=404,
        msg="Not Found",
        hdrs=None,
        fp=io.BytesIO(json.dumps(body).encode("utf-8")),
    )


def test_google_openai_compat_retries_with_fallback_model(monkeypatch) -> None:
    cfg = LLMConfig(
        backend="openai_compat",
        model="gemini-1.5-pro-latest",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        timeout_s=60,
        temperature=0,
        seed=None,
        extra_headers={"Authorization": "Bearer g-test"},
    )
    monkeypatch.setattr(llm_client, "get_llm_config", lambda user_id="default": cfg)

    attempted_models: list[str] = []

    def fake_make_request(url: str, payload: dict, headers: dict[str, str]) -> bytes:
        attempted_models.append(payload["model"])
        if payload["model"] == "gemini-1.5-pro-latest":
            raise _http_404_model_not_found()
        return json.dumps({
            "choices": [{"message": {"content": '{"twitter_summary":"ok"}'}}]
        }).encode("utf-8")

    monkeypatch.setattr(llm_client, "_make_request", fake_make_request)

    parsed, raw = generate_json_with_debug("hi")
    assert parsed is not None
    assert parsed["twitter_summary"] == "ok"
    assert raw == '{"twitter_summary":"ok"}'
    assert attempted_models[0] == "gemini-1.5-pro-latest"
    assert any(m in attempted_models for m in ["gemini-1.5-pro", "gemini-2.0-flash"])
