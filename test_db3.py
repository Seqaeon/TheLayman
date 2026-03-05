import os
import psycopg2

os.environ["DATABASE_URL"] = "postgresql://neondb_owner:password123@ep-cool-butterfly-a5z3z.us-east-2.aws.neon.tech/neondb?sslmode=require"
conn = psycopg2.connect(os.environ["DATABASE_URL"])
conn.autocommit = True
cursor = conn.cursor()

queries = [
    """CREATE TABLE IF NOT EXISTS papers ( id TEXT PRIMARY KEY, title TEXT, authors TEXT, source TEXT, url TEXT )""",
    """CREATE TABLE IF NOT EXISTS explanations ( paper_id TEXT PRIMARY KEY, content_json TEXT NOT NULL, model_used TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(paper_id) REFERENCES papers(id) )""",
    """CREATE TABLE IF NOT EXISTS user_preferences ( user_id TEXT PRIMARY KEY, target_fields TEXT NOT NULL, priority_keywords TEXT NOT NULL, relevance_instruction TEXT NOT NULL )""",
    """CREATE TABLE IF NOT EXISTS paper_scores ( paper_id TEXT PRIMARY KEY, keyword_score INTEGER NOT NULL, llm_impact_score INTEGER NOT NULL, buzz_score REAL NOT NULL DEFAULT 0, total_score REAL NOT NULL, scored_at TEXT NOT NULL )""",
    """ALTER TABLE paper_scores ADD COLUMN buzz_score REAL NOT NULL DEFAULT 0""",
    """CREATE TABLE IF NOT EXISTS daily_feed ( date TEXT PRIMARY KEY, ranked_paper_ids_json TEXT NOT NULL )""",
    """CREATE TABLE IF NOT EXISTS users ( id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL )""",
    """CREATE TABLE IF NOT EXISTS sessions ( token TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id) )""",
    """CREATE TABLE IF NOT EXISTS llm_settings_v2 ( user_id TEXT PRIMARY KEY, provider TEXT NOT NULL, openai_key TEXT NOT NULL DEFAULT '', anthropic_key TEXT NOT NULL DEFAULT '', google_key TEXT NOT NULL DEFAULT '', openai_model TEXT NOT NULL DEFAULT '', anthropic_model TEXT NOT NULL DEFAULT '', google_model TEXT NOT NULL DEFAULT '', local_model TEXT NOT NULL DEFAULT '', local_base_url TEXT NOT NULL DEFAULT '')"""
]

for i, q in enumerate(queries):
    try:
        cursor.execute(q)
        print(f"Query {i} succeeded")
    except Exception as e:
        print(f"Query {i} failed: {e}")
