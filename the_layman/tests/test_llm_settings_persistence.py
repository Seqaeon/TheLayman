from pathlib import Path

from the_layman.backend.schemas import LlmSettings
from the_layman.database.store import Store


def test_save_llm_settings_preserves_google_provider(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(db_path=db_path)

    settings = LlmSettings(
        provider="google",
        google_key="g-test-key",
        google_model="gemini-2.0-flash",
    )
    store.save_llm_settings(settings, user_id="default")

    loaded = store.get_llm_settings(user_id="default")
    assert loaded.provider == "google"
    assert loaded.google_key == "g-test-key"
    assert loaded.google_model == "gemini-2.0-flash"


def test_save_llm_settings_preserves_local_provider(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(db_path=db_path)

    settings = LlmSettings(
        provider="local",
        local_model="llama3.2",
        local_base_url="http://127.0.0.1:11434",
    )
    store.save_llm_settings(settings, user_id="default")

    loaded = store.get_llm_settings(user_id="default")
    assert loaded.provider == "local"
    assert loaded.local_model == "llama3.2"
