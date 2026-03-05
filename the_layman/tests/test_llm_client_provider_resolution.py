from pathlib import Path

from the_layman.database.store import Store
from the_layman.backend.schemas import LlmSettings
from the_layman.pipeline.llm_client import get_llm_config


def test_db_config_falls_back_from_local_to_google_when_google_is_configured(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "test.db"
    store = Store(db_path=db_path)
    store.save_llm_settings(
        LlmSettings(
            provider="local",
            local_model="",
            google_key="g-test-key",
            google_model="gemini-2.0-flash",
        ),
        user_id="default",
    )

    import the_layman.backend.app as app_module

    monkeypatch.setattr(app_module, "STORE", store)

    cfg = get_llm_config(user_id="default")
    assert cfg is not None
    assert cfg.backend == "openai_compat"
    assert cfg.model == "gemini-2.0-flash"


def test_db_config_uses_selected_google_provider_when_configured(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "test.db"
    store = Store(db_path=db_path)
    store.save_llm_settings(
        LlmSettings(
            provider="google",
            google_key="g-test-key",
            google_model="gemini-1.5-pro",
        ),
        user_id="default",
    )

    import the_layman.backend.app as app_module

    monkeypatch.setattr(app_module, "STORE", store)

    cfg = get_llm_config(user_id="default")
    assert cfg is not None
    assert cfg.backend == "openai_compat"
    assert cfg.model == "gemini-1.5-pro"
