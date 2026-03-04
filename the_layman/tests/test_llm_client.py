from __future__ import annotations

from the_layman.pipeline.llm_client import _extract_json_object, generate_json, generate_json_with_debug, model_version_tag


def test_llm_client_without_config_returns_none(monkeypatch):
    monkeypatch.delenv("LAYMAN_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("LAYMAN_MODEL_NAME", raising=False)
    assert generate_json("hello") is None
    assert model_version_tag() == "no-model-configured"



def test_model_version_tag_includes_sampling_config(monkeypatch):
    monkeypatch.setenv("LAYMAN_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("LAYMAN_MODEL_NAME", "mymodel")
    monkeypatch.setenv("LAYMAN_MODEL_TEMPERATURE", "0.3")
    monkeypatch.setenv("LAYMAN_MODEL_SEED", "42")
    tag = model_version_tag()
    assert "t=0.3" in tag
    assert "seed=42" in tag



def test_extract_json_object_from_noisy_text():
    text = "preface... {\"core_claim\": \"x\", \"confidence_level\": \"medium\"} ...suffix"
    parsed = _extract_json_object(text)
    assert parsed is not None
    assert parsed["core_claim"] == "x"



def test_generate_json_with_debug_without_config_returns_none(monkeypatch):
    monkeypatch.delenv("LAYMAN_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("LAYMAN_MODEL_NAME", raising=False)
    parsed, raw = generate_json_with_debug("hello")
    assert parsed is None
    assert raw is None
