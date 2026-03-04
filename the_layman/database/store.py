from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from the_layman.backend.schemas import (
    DailyFeedItem,
    DailyFeedResponse,
    Explanation,
    LlmSettings,
    PaperScore,
    UserPreferences,
)
from the_layman.pipeline.ingestion import PaperContent


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    source TEXT,
                    url TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS explanations (
                    paper_id TEXT PRIMARY KEY,
                    content_json TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES papers(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    target_fields TEXT NOT NULL,
                    priority_keywords TEXT NOT NULL,
                    relevance_instruction TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_scores (
                    paper_id TEXT PRIMARY KEY,
                    keyword_score INTEGER NOT NULL,
                    llm_impact_score INTEGER NOT NULL,
                    buzz_score REAL NOT NULL DEFAULT 0,
                    total_score REAL NOT NULL,
                    scored_at TEXT NOT NULL
                )
                """
            )
            # Migrate existing DBs that lack the buzz_score column
            try:
                conn.execute("ALTER TABLE paper_scores ADD COLUMN buzz_score REAL NOT NULL DEFAULT 0")
            except Exception:
                pass  # Column already exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_feed (
                    date TEXT PRIMARY KEY,
                    ranked_paper_ids_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_settings_v2 (
                user_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                openai_key TEXT NOT NULL DEFAULT '',
                anthropic_key TEXT NOT NULL DEFAULT '',
                google_key TEXT NOT NULL DEFAULT '',
                openai_model TEXT NOT NULL DEFAULT '',
                anthropic_model TEXT NOT NULL DEFAULT '',
                google_model TEXT NOT NULL DEFAULT '',
                local_model TEXT NOT NULL DEFAULT '',
                local_base_url TEXT NOT NULL DEFAULT ''
            )
            """
        )

    def save_paper(self, paper: PaperContent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO papers (id, title, authors, source, url)
                VALUES (?, ?, ?, ?, ?)
                """,
                (paper.paper_id, paper.title, json.dumps(paper.authors), paper.source, paper.url),
            )

    def save_explanation(
        self,
        paper_id: str,
        explanation: Explanation,
        model_used: str | None = None,
        runtime_model: str | None = None,
    ) -> None:
        """Backwards-compatible save supporting both `model_used` and `runtime_model` names."""
        resolved_model = model_used or runtime_model or "unknown"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO explanations (paper_id, content_json, model_used, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    paper_id,
                    explanation.model_dump_json(),
                    resolved_model,
                    explanation.generated_timestamp,
                ),
            )

    def get_explanation(
        self,
        paper_id: str,
        model_used: str | None = None,
        runtime_model: str | None = None,
    ) -> Explanation | None:
        """Backwards-compatible read supporting both `model_used` and `runtime_model` names."""
        resolved_model = model_used or runtime_model
        with self._connect() as conn:
            if resolved_model is None:
                row = conn.execute(
                    "SELECT content_json FROM explanations WHERE paper_id = ?", (paper_id,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT content_json FROM explanations WHERE paper_id = ? AND model_used = ?",
                    (paper_id, resolved_model),
                ).fetchone()
            if not row:
                return None
            return Explanation.parse_raw(row["content_json"])

    def feed(self, limit: int = 10) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT p.id, p.title, p.source, e.content_json
                FROM papers p
                JOIN explanations e ON p.id = e.paper_id
                ORDER BY e.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def get_user_preferences(self, user_id: str = "default") -> UserPreferences:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT target_fields, priority_keywords, relevance_instruction FROM user_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                return UserPreferences(user_id=user_id)
            return UserPreferences(
                user_id=user_id,
                target_fields=json.loads(row["target_fields"]),
                priority_keywords=json.loads(row["priority_keywords"]),
                relevance_instruction=row["relevance_instruction"],
            )

    def save_user_preferences(self, prefs: UserPreferences) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_preferences (user_id, target_fields, priority_keywords, relevance_instruction)
                VALUES (?, ?, ?, ?)
                """,
                (
                    prefs.user_id,
                    json.dumps(prefs.target_fields),
                    json.dumps(prefs.priority_keywords),
                    prefs.relevance_instruction,
                ),
            )

    def save_paper_scores(self, scores: list[PaperScore]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO paper_scores (paper_id, keyword_score, llm_impact_score, buzz_score, total_score, scored_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (s.paper_id, s.keyword_score, s.llm_impact_score, s.buzz_score, s.total_score, s.scored_at)
                    for s in scores
                ],
            )

    def save_daily_feed(self, date: str, items: list[DailyFeedItem]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_feed (date, ranked_paper_ids_json)
                VALUES (?, ?)
                """,
                (date, json.dumps([item.dict() for item in items])),
            )

    def get_daily_feed(self, date: str) -> DailyFeedResponse | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ranked_paper_ids_json FROM daily_feed WHERE date = ?", (date,)
            ).fetchone()
            if not row:
                return None
            items_raw = json.loads(row["ranked_paper_ids_json"])
            items = [DailyFeedItem.parse_obj(raw) for raw in items_raw]
            return DailyFeedResponse(date=date, items=items)

    def get_llm_settings(self, user_id: str = "default") -> LlmSettings:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT provider, openai_key, anthropic_key, google_key, 
                          openai_model, anthropic_model, google_model, 
                          local_model, local_base_url 
                   FROM llm_settings_v2 WHERE user_id = ?""", (user_id,)
            ).fetchone()
            if not row:
                return LlmSettings()
            return LlmSettings(
                provider=row["provider"],
                openai_key=row["openai_key"],
                anthropic_key=row["anthropic_key"],
                google_key=row["google_key"],
                openai_model=row["openai_model"],
                anthropic_model=row["anthropic_model"],
                google_model=row["google_model"],
                local_model=row["local_model"],
                local_base_url=row["local_base_url"],
            )

    def save_llm_settings(self, settings: LlmSettings, user_id: str = "default") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_settings_v2 
                (user_id, provider, openai_key, anthropic_key, google_key, 
                 openai_model, anthropic_model, google_model, local_model, local_base_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, settings.provider, settings.openai_key, settings.anthropic_key, 
                 settings.google_key, settings.openai_model, settings.anthropic_model, 
                 settings.google_model, settings.local_model, settings.local_base_url),
            )

    # ----------------------------------------------------------------------
    # Auth & Users
    # ----------------------------------------------------------------------

    def create_user(self, user_id: str, username: str, password_hash: str) -> bool:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
                    (user_id, username, password_hash)
                )
            return True
        except Exception:
            return False  # e.g. UNIQUE constraint failed

    def get_user_by_username(self, username: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,)).fetchone()
            if row:
                return dict(row)
        return None

    def create_session(self, token: str, user_id: str, expires_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at)
            )

    def get_user_by_session(self, token: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username 
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = ? AND s.expires_at > datetime('now')
                """,
                (token,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def delete_session(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
