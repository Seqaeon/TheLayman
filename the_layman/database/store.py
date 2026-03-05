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


import os
import re

try:
    import psycopg2
    from psycopg2.extras import DictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


class Store:
    def __init__(self, db_path: Path | None = None, db_url: str | None = None):
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.is_postgres = self.db_url and self.db_url.startswith("postgres")
        
        if self.is_postgres and not HAS_PSYCOPG2:
            raise RuntimeError("DATABASE_URL indicates Postgres but psycopg2-binary is not installed.")

        if not self.is_postgres:
            if not db_path:
                raise ValueError("db_path must be provided if DATABASE_URL is not set.")
            self.db_path = db_path
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.db_path = None
            
        self._init_db()

    def _connect(self):
        """Returns a managed connection object depending on the backend."""
        if self.is_postgres:
            conn = psycopg2.connect(self.db_url, cursor_factory=DictCursor)
            conn.autocommit = True
            return conn
        else:
            # We guaranteed self.db_path is not None above if not self.is_postgres
            conn = sqlite3.connect(self.db_path) # type: ignore
            conn.row_factory = sqlite3.Row
            return conn

    def _execute(self, conn, query: str, params: tuple = ()) -> sqlite3.Cursor | any:
        """Executes a query, translating SQLite syntax to Postgres syntax if necessary."""
        if self.is_postgres:
            # Simple dialect translation
            # 1. ? -> %s
            pg_query = query.replace("?", "%s")
            
            # 2. SQLite INSERT OR REPLACE -> Postgres INSERT ON CONFLICT DO UPDATE
            if "INSERT OR REPLACE INTO papers" in pg_query:
                pg_query = pg_query.replace("INSERT OR REPLACE INTO papers (id, title, authors, source, url)", 
                                            "INSERT INTO papers (id, title, authors, source, url)")
                pg_query += " ON CONFLICT (id) DO UPDATE SET title=EXCLUDED.title, authors=EXCLUDED.authors, source=EXCLUDED.source, url=EXCLUDED.url"
                
            elif "INSERT OR REPLACE INTO explanations" in pg_query:
                pg_query = pg_query.replace("INSERT OR REPLACE INTO explanations (paper_id, content_json, model_used, created_at)",
                                            "INSERT INTO explanations (paper_id, content_json, model_used, created_at)")
                pg_query += " ON CONFLICT (paper_id) DO UPDATE SET content_json=EXCLUDED.content_json, model_used=EXCLUDED.model_used, created_at=EXCLUDED.created_at"
                
            elif "INSERT OR REPLACE INTO user_preferences" in pg_query:
                pg_query = pg_query.replace("INSERT OR REPLACE INTO user_preferences (user_id, target_fields, priority_keywords, relevance_instruction)",
                                            "INSERT INTO user_preferences (user_id, target_fields, priority_keywords, relevance_instruction)")
                pg_query += " ON CONFLICT (user_id) DO UPDATE SET target_fields=EXCLUDED.target_fields, priority_keywords=EXCLUDED.priority_keywords, relevance_instruction=EXCLUDED.relevance_instruction"
                
            elif "INSERT OR REPLACE INTO paper_scores" in pg_query:
                pg_query = pg_query.replace("INSERT OR REPLACE INTO paper_scores (paper_id, keyword_score, llm_impact_score, buzz_score, total_score, scored_at)",
                                            "INSERT INTO paper_scores (paper_id, keyword_score, llm_impact_score, buzz_score, total_score, scored_at)")
                pg_query += " ON CONFLICT (paper_id) DO UPDATE SET keyword_score=EXCLUDED.keyword_score, llm_impact_score=EXCLUDED.llm_impact_score, buzz_score=EXCLUDED.buzz_score, total_score=EXCLUDED.total_score, scored_at=EXCLUDED.scored_at"
                
            elif "INSERT OR REPLACE INTO daily_feed" in pg_query:
                pg_query = pg_query.replace("INSERT OR REPLACE INTO daily_feed (date, ranked_paper_ids_json)",
                                            "INSERT INTO daily_feed (date, ranked_paper_ids_json)")
                pg_query += " ON CONFLICT (date) DO UPDATE SET ranked_paper_ids_json=EXCLUDED.ranked_paper_ids_json"
                
            elif "INSERT OR REPLACE INTO llm_settings_v2" in pg_query:
                pg_query = pg_query.replace("INSERT OR REPLACE INTO llm_settings_v2 \n                (user_id, provider, openai_key, anthropic_key, google_key, \n                 openai_model, anthropic_model, google_model, local_model, local_base_url)",
                                            "INSERT INTO llm_settings_v2 (user_id, provider, openai_key, anthropic_key, google_key, openai_model, anthropic_model, google_model, local_model, local_base_url)")
                pg_query += " ON CONFLICT (user_id) DO UPDATE SET provider=EXCLUDED.provider, openai_key=EXCLUDED.openai_key, anthropic_key=EXCLUDED.anthropic_key, google_key=EXCLUDED.google_key, openai_model=EXCLUDED.openai_model, anthropic_model=EXCLUDED.anthropic_model, google_model=EXCLUDED.google_model, local_model=EXCLUDED.local_model, local_base_url=EXCLUDED.local_base_url"
            
            cursor = conn.cursor()
            cursor.execute(pg_query, params)
            return cursor
        else:
            return conn.execute(query, params)

    def _executemany(self, conn, query: str, params_list: list) -> None:
        if self.is_postgres:
            pg_query = query.replace("?", "%s")
            # Currently only used for paper_scores
            if "INSERT OR REPLACE INTO paper_scores" in pg_query:
                pg_query = pg_query.replace("INSERT OR REPLACE INTO paper_scores (paper_id, keyword_score, llm_impact_score, buzz_score, total_score, scored_at)",
                                            "INSERT INTO paper_scores (paper_id, keyword_score, llm_impact_score, buzz_score, total_score, scored_at)")
                pg_query += " ON CONFLICT (paper_id) DO UPDATE SET keyword_score=EXCLUDED.keyword_score, llm_impact_score=EXCLUDED.llm_impact_score, buzz_score=EXCLUDED.buzz_score, total_score=EXCLUDED.total_score, scored_at=EXCLUDED.scored_at"
            cursor = conn.cursor()
            cursor.executemany(pg_query, params_list)
        else:
            conn.executemany(query, params_list)

    def _init_db(self) -> None:
        with self._connect() as conn:
            # Postgres needs slightly different schema types or ignores SQLite specific pragmas
            # We keep it generic. SQLite TEXT and Postgres TEXT work the same.
            self._execute(conn,
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
            self._execute(conn,
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
            self._execute(conn,
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    target_fields TEXT NOT NULL,
                    priority_keywords TEXT NOT NULL,
                    relevance_instruction TEXT NOT NULL
                )
                """
            )
            self._execute(conn,
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
            # Migrate existing SQLite DBs that lack the buzz_score column
            if not self.is_postgres:
                try:
                    conn.execute("ALTER TABLE paper_scores ADD COLUMN buzz_score REAL NOT NULL DEFAULT 0")
                except Exception:
                    pass  # Column already exists
            else:
                 try:
                     # For Postgres, if autocommit is True, the exception shouldn't break the whole script, but 
                     # we can bypass the error entirely by checking if the column exists first:
                     row = self._execute(conn, "SELECT column_name FROM information_schema.columns WHERE table_name='paper_scores' AND column_name='buzz_score'").fetchone()
                     if not row:
                         self._execute(conn, "ALTER TABLE paper_scores ADD COLUMN buzz_score REAL NOT NULL DEFAULT 0")
                 except Exception as e:
                     print(f"Postgres migration skipped: {e}")

            self._execute(conn,
                """
                CREATE TABLE IF NOT EXISTS daily_feed (
                    date TEXT PRIMARY KEY,
                    ranked_paper_ids_json TEXT NOT NULL
                )
                """
            )
            self._execute(conn,
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                )
                """
            )
            self._execute(conn,
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            self._execute(conn,
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
            self._execute(conn,
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
        resolved_model = model_used or runtime_model or "unknown"
        with self._connect() as conn:
            self._execute(conn,
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
        resolved_model = model_used or runtime_model
        with self._connect() as conn:
            if resolved_model is None:
                row = self._execute(conn,
                    "SELECT content_json FROM explanations WHERE paper_id = ?", (paper_id,)
                ).fetchone()
            else:
                row = self._execute(conn,
                    "SELECT content_json FROM explanations WHERE paper_id = ? AND model_used = ?",
                    (paper_id, resolved_model),
                ).fetchone()
            if not row:
                return None
            return Explanation.parse_raw(row["content_json"])

    def feed(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = self._execute(conn,
                """
                SELECT p.id, p.title, p.source, e.content_json
                FROM papers p
                JOIN explanations e ON p.id = e.paper_id
                ORDER BY e.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_user_preferences(self, user_id: str = "default") -> UserPreferences:
        with self._connect() as conn:
            row = self._execute(conn,
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
            self._execute(conn,
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
            self._executemany(conn,
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
            self._execute(conn,
                """
                INSERT OR REPLACE INTO daily_feed (date, ranked_paper_ids_json)
                VALUES (?, ?)
                """,
                (date, json.dumps([item.dict() for item in items])),
            )

    def get_daily_feed(self, date: str) -> DailyFeedResponse | None:
        with self._connect() as conn:
            row = self._execute(conn,
                "SELECT ranked_paper_ids_json FROM daily_feed WHERE date = ?", (date,)
            ).fetchone()
            if not row:
                return None
            items_raw = json.loads(row["ranked_paper_ids_json"])
            items = [DailyFeedItem.parse_obj(raw) for raw in items_raw]
            return DailyFeedResponse(date=date, items=items)

    def get_llm_settings(self, user_id: str = "default") -> LlmSettings:
        with self._connect() as conn:
            row = self._execute(conn,
                """SELECT provider, openai_key, anthropic_key, google_key, 
                          openai_model, anthropic_model, google_model, 
                          local_model, local_base_url 
                   FROM llm_settings_v2 WHERE user_id = ?""", (user_id,)
            ).fetchone()
            if not row:
                return LlmSettings()
            provider_val = row["provider"]
            if provider_val == "local":
                provider_val = "openai"
                
            return LlmSettings(
                provider=provider_val,
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
            self._execute(conn,
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
                self._execute(conn,
                    "INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
                    (user_id, username, password_hash)
                )
            return True
        except Exception:
            return False  # e.g. UNIQUE constraint failed

    def get_user_by_username(self, username: str) -> dict | None:
        with self._connect() as conn:
            row = self._execute(conn, "SELECT id, username, password_hash FROM users WHERE username = ?", (username,)).fetchone()
            if row:
                return dict(row)
        return None

    def create_session(self, token: str, user_id: str, expires_at: str) -> None:
        with self._connect() as conn:
            self._execute(conn,
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at)
            )

    def get_user_by_session(self, token: str) -> dict | None:
        with self._connect() as conn:
            # Postgres doesn't have datetime('now'), we map it if needed
            if self.is_postgres:
                query = "SELECT u.id, u.username FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ? AND s.expires_at::TIMESTAMP > CURRENT_TIMESTAMP"
            else:
                query = "SELECT u.id, u.username FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ? AND s.expires_at > datetime('now')"

            row = self._execute(conn, query, (token,)).fetchone()
            if row:
                return dict(row)
        return None

    def delete_session(self, token: str) -> None:
        with self._connect() as conn:
            self._execute(conn, "DELETE FROM sessions WHERE token = ?", (token,))
