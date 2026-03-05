from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, validator


UNKNOWN_VALUE = "unknown"


class WhyItMatters(BaseModel):
    who_it_affects: str = UNKNOWN_VALUE
    problems_solved: str = UNKNOWN_VALUE
    timeline_of_impact: str = UNKNOWN_VALUE
    limitations: str = UNKNOWN_VALUE


class Explanation(BaseModel):
    core_claim: str = UNKNOWN_VALUE
    twitter_summary: str = UNKNOWN_VALUE
    coffee_chat: str = UNKNOWN_VALUE
    deep_dive: str = UNKNOWN_VALUE
    why_it_matters: WhyItMatters = Field(default_factory=WhyItMatters)
    confidence_level: Literal["low", "medium", "high", "unknown"] = "unknown"
    original_paper_link: str = UNKNOWN_VALUE
    sources_used: List[str] = Field(default_factory=list)
    generated_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @validator("twitter_summary")
    @classmethod
    def ensure_twitter_has_content(cls, value: str) -> str:
        return value if value.strip() else UNKNOWN_VALUE


class ExplainRequest(BaseModel):
    doi: str | None = None
    arxiv_url: str | None = None
    regenerate: bool = False


class ExplainResponse(BaseModel):
    paper_id: str
    explanation: Explanation
    cached: bool
    runtime_model: str = "unknown"
    prompt_used: str = "unknown"
    raw_model_output: dict[str, Any] | str = "unknown"


class FeedItem(BaseModel):
    paper_id: str
    title: str
    field: str
    relevance_reason: str
    explanation: Explanation


class FeedResponse(BaseModel):
    items: List[FeedItem]


class UserPreferences(BaseModel):
    user_id: str = "default"
    target_fields: List[str] = Field(default_factory=list)
    priority_keywords: List[str] = Field(default_factory=list)
    relevance_instruction: str = "Prioritize papers that offer immediate, tangible improvements to consumer technology, medicine, or software efficiency over purely theoretical mathematical proofs."


class LlmSettings(BaseModel):
    provider: Literal["local", "openai", "anthropic", "google"] = "openai"
    openai_key: str = ""
    anthropic_key: str = ""
    google_key: str = ""
    openai_model: str = ""
    anthropic_model: str = ""
    google_model: str = ""
    local_model: str = ""
    local_base_url: str = ""

class User(BaseModel):
    id: str
    username: str
    password_hash: str


class PaperScore(BaseModel):
    paper_id: str
    keyword_score: int = 0
    llm_impact_score: int = 0
    buzz_score: float = 0.0
    total_score: float = 0.0
    scored_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DailyFeedItem(BaseModel):
    paper_id: str
    title: str
    field: str
    impact_score: int | str
    buzz_score: float = 0.0
    abstract_preview: str


class DailyFeedResponse(BaseModel):
    date: str
    items: List[DailyFeedItem]
