from fastapi.testclient import TestClient

from the_layman.backend.app import app


client = TestClient(app)


def test_explain_and_get_cached():
    res = client.post("/explain", json={"doi": "10.1000/demo", "regenerate": False})
    assert res.status_code == 200
    data = res.json()
    assert "paper_id" in data

    paper_id = data["paper_id"]
    cached_res = client.get(f"/paper/{paper_id}")
    assert cached_res.status_code == 200
    assert cached_res.json()["cached"] is True


def test_feed_endpoint():
    client.post("/explain", json={"doi": "10.1000/feed", "regenerate": True})
    res = client.get("/feed")
    assert res.status_code == 200
    assert "items" in res.json()


def test_root_serves_frontend():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")


def test_favicon_no_content():
    res = client.get("/favicon.ico")
    assert res.status_code == 204


def test_explain_requires_model_or_explicit_fallback(monkeypatch):
    monkeypatch.delenv("LAYMAN_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("LAYMAN_MODEL_NAME", raising=False)
    monkeypatch.setenv("LAYMAN_ALLOW_GROUNDED_FALLBACK", "0")
    res = client.post("/explain", json={"doi": "10.1000/demo", "regenerate": True})
    assert res.status_code == 503


def test_get_llm_settings_default():
    """GET /api/llm_settings returns a valid LlmSettings object with provider=local."""
    res = client.get("/api/llm_settings")
    assert res.status_code == 200
    data = res.json()
    assert "provider" in data
    assert data["provider"] == "local"
    assert "model_name" in data
    assert "api_key" in data
    assert "base_url" in data


def test_save_and_retrieve_llm_settings():
    """POST /api/llm_settings then GET round-trips correctly."""
    payload = {
        "provider": "openai",
        "model_name": "gpt-4o",
        "api_key": "sk-test-12345",
        "base_url": "",
    }
    post_res = client.post("/api/llm_settings", json=payload)
    assert post_res.status_code == 200
    saved = post_res.json()
    assert saved["provider"] == "openai"
    assert saved["model_name"] == "gpt-4o"

    get_res = client.get("/api/llm_settings")
    assert get_res.status_code == 200
    fetched = get_res.json()
    assert fetched["provider"] == "openai"
    assert fetched["model_name"] == "gpt-4o"
    assert fetched["api_key"] == "sk-test-12345"
