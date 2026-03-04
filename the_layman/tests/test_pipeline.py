from pathlib import Path

from the_layman.database.store import Store
from the_layman.pipeline.generator import build_explanation
from the_layman.pipeline.ingestion import ingest_doi


def test_explanation_schema_fields_exist():
    paper = ingest_doi("10.1000/test")
    explanation = build_explanation(paper)
    payload = explanation.model_dump()

    required = {
        "core_claim",
        "twitter_summary",
        "coffee_chat",
        "deep_dive",
        "why_it_matters",
        "confidence_level",
        "original_paper_link",
        "sources_used",
        "generated_timestamp",
    }
    assert required.issubset(payload.keys())


def test_cache_roundtrip(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite")
    paper = ingest_doi("10.1000/cache")
    explanation = build_explanation(paper)

    store.save_paper(paper)
    store.save_explanation(paper.paper_id, explanation, model_used="test-model")

    loaded = store.get_explanation(paper.paper_id)
    assert loaded is not None
    assert loaded.core_claim == explanation.core_claim



def test_cache_model_version_miss(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite")
    paper = ingest_doi("10.1000/cache-model")
    explanation = build_explanation(paper)
    store.save_paper(paper)
    store.save_explanation(paper.paper_id, explanation, model_used="model-a")
    assert store.get_explanation(paper.paper_id, model_used="model-b") is None


def test_arxiv_ingestion_html_fallback(monkeypatch):
    from the_layman.pipeline import ingestion

    xml = b"<?xml version='1.0' encoding='UTF-8'?><feed xmlns='http://www.w3.org/2005/Atom'></feed>"
    html = b"<html><head><meta name=\"citation_title\" content=\"Test Paper\"/><meta name=\"citation_abstract\" content=\"This is an abstract from abs page.\"/></head></html>"

    def fake_get(url: str):
        if "export.arxiv.org" in url:
            return xml
        if "/abs/" in url:
            return html
        raise ValueError("no pdf in test")

    monkeypatch.setattr(ingestion, "_http_get", fake_get)
    paper = ingestion.ingest_arxiv("https://arxiv.org/abs/2602.23330")
    assert paper.title == "Test Paper"
    assert "abstract from abs page" in paper.abstract.lower()


def test_llm_unknown_output_falls_back_to_grounded(monkeypatch):
    from the_layman.pipeline import generator

    monkeypatch.setattr(
        generator,
        "generate_json_with_debug",
        lambda _prompt: ({
            "core_claim": "unknown",
            "twitter_summary": "unknown",
            "coffee_chat": "unknown",
            "deep_dive": "unknown",
            "why_it_matters": {
                "who_it_affects": "unknown",
                "problems_solved": "unknown",
                "timeline_of_impact": "unknown",
                "limitations": "unknown",
            },
            "confidence_level": "unknown",
        }, "raw")
    )

    paper = ingest_doi("10.1000/test")
    explanation = generator.build_explanation(paper)
    assert explanation.twitter_summary != "unknown"
    assert explanation.why_it_matters.who_it_affects != "unknown"
    assert explanation.why_it_matters.problems_solved != "unknown"
    assert explanation.why_it_matters.timeline_of_impact != "unknown"



def test_deep_dive_is_detailed():
    paper = ingest_doi("10.1000/deepdive")
    explanation = build_explanation(paper)
    assert len(explanation.deep_dive) > 800
