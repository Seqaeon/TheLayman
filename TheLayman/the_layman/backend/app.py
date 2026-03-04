from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
import secrets

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Cookie, Depends, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from the_layman.backend.schemas import (
    DailyFeedResponse,
    ExplainRequest,
    ExplainResponse,
    FeedItem,
    FeedResponse,
    LlmSettings,
    UserPreferences,
)
from the_layman.database.store import Store
from the_layman.pipeline.generator import PROMPT_VERSION, build_explanation_with_debug
from the_layman.pipeline.ingestion import PaperContent, ingest_arxiv, ingest_doi, ingest_pdf
from the_layman.pipeline.llm_client import model_version_tag
from the_layman.backend.auth import hash_password, verify_password, generate_session_token, generate_session_expiry


BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "cache" / "uploads"

# --- DB path: configurable for cloud (Render persistent disk) ---
_db_path_env = os.environ.get("DB_PATH")
if _db_path_env:
    DB_PATH = Path(_db_path_env)
else:
    DB_PATH = BASE_DIR.parent / "cache" / "the_layman.db"

STORE = Store(DB_PATH)

app = FastAPI(title="THE LAYMAN API")


# ── Startup: admin seeding + APScheduler ─────────────────────────────────────

def _seed_admin() -> None:
    """Create the admin user from env vars if no users exist yet."""
    username = os.environ.get("ADMIN_USERNAME", "").strip()
    password = os.environ.get("ADMIN_PASSWORD", "").strip()
    if not username or not password:
        return
    with STORE._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count > 0:
            return  # already seeded
    pw_hash = hash_password(password)
    STORE.create_user("default", username, pw_hash)
    print(f"[startup] Admin user '{username}' created.")


def _run_daily_feed() -> None:
    from the_layman.pipeline.daily_feed import generate_daily_feed
    try:
        print("[scheduler] Running daily feed generation...")
        generate_daily_feed(STORE)
        print("[scheduler] Daily feed done.")
    except Exception as exc:
        print(f"[scheduler] Daily feed error: {exc}")


@app.on_event("startup")
async def startup_event() -> None:
    _seed_admin()
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        hour = int(os.environ.get("FEED_SCHEDULE_HOUR", "6"))
        scheduler = BackgroundScheduler()
        scheduler.add_job(_run_daily_feed, CronTrigger(hour=hour, minute=0), id="daily_feed")
        scheduler.start()
        print(f"[startup] APScheduler started — daily feed at {hour:02d}:00 UTC")
    except ImportError:
        print("[startup] apscheduler not installed — daily feed scheduling disabled")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(BASE_DIR / "frontend" / "feed.html")

@app.get("/view")
@app.get("/custom")
def custom_explain() -> FileResponse:
    return FileResponse(BASE_DIR / "frontend" / "index.html")

@app.get("/login.html")
def login_page() -> FileResponse:
    return FileResponse(BASE_DIR / "frontend" / "login.html")

@app.get("/feed_ui")
def feed_ui_redirect() -> FileResponse:
    # Keep this for backward compatibility if needed, or redirect
    return FileResponse(BASE_DIR / "frontend" / "feed.html")

@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)




def _require_model_ready() -> None:
    allow_fallback = os.getenv("LAYMAN_ALLOW_GROUNDED_FALLBACK", "0") == "1"
    if model_version_tag() == "no-model-configured" and not allow_fallback:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model configured. Set LAYMAN_MODEL_BACKEND and LAYMAN_MODEL_NAME "
                "(or set LAYMAN_ALLOW_GROUNDED_FALLBACK=1 for fallback mode)."
            ),
        )

def _hydrate(req: ExplainRequest, pdf_path: Path | None = None) -> PaperContent:
    if req.doi:
        return ingest_doi(req.doi)
    if req.arxiv_url:
        url = str(req.arxiv_url)
        if not url.startswith("http"):
            # Handle arxiv:ID or just ID
            clean_id = url.replace("arxiv:", "")
            # USE THE ABSTRACT URL as the canonical starting point
            url = f"https://arxiv.org/abs/{clean_id}"
        return ingest_arxiv(url)
    if pdf_path:
        return ingest_pdf(pdf_path)
    raise HTTPException(status_code=400, detail="Provide doi, arxiv_url, or pdf upload")


@app.post("/api/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest) -> ExplainResponse:
    _require_model_ready()
    paper = _hydrate(req)
    active_model = f"{model_version_tag()}|{PROMPT_VERSION}"
    if not req.regenerate:
        cached = STORE.get_explanation(paper.paper_id, runtime_model=active_model)
        if cached:
            return ExplainResponse(paper_id=paper.paper_id, explanation=cached, cached=True, runtime_model=active_model, prompt_used="unknown", raw_model_output="unknown")

    explanation, raw_model_output, prompt_used, raw_model_text = build_explanation_with_debug(paper)
    STORE.save_paper(paper)
    STORE.save_explanation(paper.paper_id, explanation, runtime_model=active_model)

    return ExplainResponse(
        paper_id=paper.paper_id,
        explanation=explanation,
        cached=False,
        runtime_model=active_model,
        prompt_used=prompt_used,
        raw_model_output=(raw_model_output or raw_model_text or "unknown"),
    )


@app.post("/api/explain/pdf", response_model=ExplainResponse)
async def explain_pdf(
    pdf: UploadFile = File(...),
    regenerate: bool = Query(default=False),
) -> ExplainResponse:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = UPLOAD_DIR / pdf.filename
    pdf_path.write_bytes(await pdf.read())

    _require_model_ready()
    paper = _hydrate(ExplainRequest(regenerate=regenerate), pdf_path)
    active_model = f"{model_version_tag()}|{PROMPT_VERSION}"
    if not regenerate:
        cached = STORE.get_explanation(paper.paper_id, runtime_model=active_model)
        if cached:
            return ExplainResponse(paper_id=paper.paper_id, explanation=cached, cached=True, runtime_model=active_model, prompt_used="unknown", raw_model_output="unknown")

    explanation, raw_model_output, prompt_used, raw_model_text = build_explanation_with_debug(paper)
    STORE.save_paper(paper)
    STORE.save_explanation(paper.paper_id, explanation, runtime_model=active_model)

    return ExplainResponse(
        paper_id=paper.paper_id,
        explanation=explanation,
        cached=False,
        runtime_model=active_model,
        prompt_used=prompt_used,
        raw_model_output=(raw_model_output or raw_model_text or "unknown"),
    )


@app.get("/paper/{paper_id}", response_model=ExplainResponse)
def get_paper(paper_id: str) -> ExplainResponse:
    explanation = STORE.get_explanation(paper_id)
    if not explanation:
        raise HTTPException(status_code=404, detail="Paper not found")
    return ExplainResponse(
        paper_id=paper_id,
        explanation=explanation,
        cached=True,
        runtime_model="unknown",
        prompt_used="unknown",
        raw_model_output="unknown",
    )


@app.get("/feed", response_model=FeedResponse)
def feed() -> FeedResponse:
    rows = STORE.feed(limit=20)
    seen_fields: set[str] = set()
    items: list[FeedItem] = []
    for row in rows:
        field = row["source"]
        if field in seen_fields:
            continue
        seen_fields.add(field)
        explanation = json.loads(row["content_json"])
        items.append(
            FeedItem(
                paper_id=row["id"],
                title=row["title"] or "unknown",
                field=field,
                relevance_reason="Selected for clear real-world relevance.",
                explanation=explanation,
            )
        )
    return FeedResponse(items=items)


# --- Auth & Users ---

class AuthRequest(BaseModel):
    username: str
    password: str

def get_current_user(session_token: str | None = Cookie(None)) -> dict:
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = STORE.get_user_by_session(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session")
    return user

@app.post("/api/register")
def register(req: AuthRequest, response: Response) -> dict:
    if len(req.username) < 3 or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Username>=3, Password>=6 chars required")
    
    with STORE._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count >= 1:
            raise HTTPException(status_code=400, detail="Registration is locked: an admin already exists.")

    user_id = "default"
    pw_hash = hash_password(req.password)
    if not STORE.create_user(user_id, req.username, pw_hash):
        raise HTTPException(status_code=400, detail="Username already taken")
    
    token = generate_session_token()
    STORE.create_session(token, user_id, generate_session_expiry())
    response.set_cookie(key="session_token", value=token, httponly=True, max_age=30*24*3600)
    return {"message": "Registered and logged in"}

@app.post("/api/login")
def login(req: AuthRequest, response: Response) -> dict:
    user = STORE.get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = generate_session_token()
    STORE.create_session(token, user["id"], generate_session_expiry())
    response.set_cookie(key="session_token", value=token, httponly=True, max_age=30*24*3600)
    return {"message": "Logged in"}

@app.post("/api/logout")
def logout(response: Response, session_token: str | None = Cookie(None)) -> dict:
    if session_token:
        STORE.delete_session(session_token)
    response.delete_cookie("session_token")
    return {"message": "Logged out"}


# --- Protected Settings ---

@app.get("/api/settings", response_model=UserPreferences)
def get_settings(user: dict = Depends(get_current_user)) -> UserPreferences:
    return STORE.get_user_preferences(user_id=user["id"])

@app.post("/api/settings", response_model=UserPreferences)
def update_settings(prefs: UserPreferences, user: dict = Depends(get_current_user)) -> UserPreferences:
    prefs.user_id = user["id"]
    STORE.save_user_preferences(prefs)
    return prefs

@app.get("/api/daily_feed", response_model=DailyFeedResponse)
def get_daily_feed(date: str) -> DailyFeedResponse:
    feed = STORE.get_daily_feed(date)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not generated for this date")
    return feed

@app.post("/api/trigger_feed_update")
async def trigger_feed_update() -> dict:
    from the_layman.pipeline.daily_feed import generate_daily_feed
    # Background this ideally, but for now run synchronously for simplicity in testing
    # In a real app, use BackgroundTasks
    generate_daily_feed(STORE)
    return {"status": "success", "message": "Daily feed updated"}


@app.get("/api/llm_settings", response_model=LlmSettings)
def get_llm_settings(user: dict = Depends(get_current_user)) -> LlmSettings:
    return STORE.get_llm_settings(user_id=user["id"])


@app.post("/api/llm_settings", response_model=LlmSettings)
def update_llm_settings(settings: LlmSettings, user: dict = Depends(get_current_user)) -> LlmSettings:
    STORE.save_llm_settings(settings, user_id=user["id"])
    return settings
