# TheLayman

A web app that explains academic papers in plain language — Twitter summary, coffee-chat explanation, and deep dive — powered by your choice of LLM.

Supports **OpenAI**, **Anthropic (Claude)**, **Google AI (Gemini)**, and **local Ollama** models.

---

## Features

- 📄 **Paper ingestion** — arXiv URL, DOI, or uploaded PDF
- 🤖 **Multi-model support** — configure separate API keys per provider; they never overwrite each other
- 📰 **Daily Discovery Feed** — ranked papers from arXiv based on your field and keyword preferences
- 🔐 **Login system** — single-admin auth with scrypt-hashed passwords and HTTP-only session cookies
- ☁️ **Cloud-deployable** — one-click deploy to Render.com with a persistent disk for SQLite

---

## Quick start (local)

```bash
git clone <your-repo-url>
cd TheLayman

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app:app --reload
```

Open [http://localhost:8000/login.html](http://localhost:8000/login.html), register your account, then configure your LLM in **Model Settings**.

> **Note:** On first run with no `ADMIN_USERNAME` env var, registration is open until one account is created. After that, registration is locked.

---

## Using a local model (Ollama)

1. Install [Ollama](https://ollama.com) and pull a model: `ollama pull llama3.2`
2. Log in → open ⚙ Model Settings → choose **Local (Ollama)**
3. Enter your model name (e.g. `llama3.2`) and Ollama URL (default: `http://127.0.0.1:11434`)
4. Save

---

## Using a cloud model (OpenAI / Claude / Gemini)

1. Log in → open ⚙ Model Settings → choose your provider
2. Paste your API key and model name
3. Save — each provider's key is stored independently

---

## Deploy to Render.com

See [walkthrough.md](https://github.com) or the summary below:

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service** → connect your repo (detects `render.yaml` automatically)
3. In Render's **Environment** settings, add:
   - `ADMIN_USERNAME` — your admin username
   - `ADMIN_PASSWORD` — your admin password
4. Deploy. Your account is auto-created on first boot.

The daily feed runs at **6am UTC** by default (change with `FEED_SCHEDULE_HOUR` env var).

> ⚠️ **Local Ollama does not work on cloud** — use OpenAI, Anthropic, or Google AI when deployed.

---

## Project layout

```
the_layman/
  backend/      # FastAPI app, auth, schemas
  frontend/     # HTML/CSS/JS (feed.html, index.html, login.html)
  pipeline/     # LLM client, ingestion, generation, daily feed
  database/     # SQLite store
render.yaml     # Render.com deploy config
Procfile        # Process startup command
.env.example    # All available environment variables
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `cache/the_layman.db` | Path to SQLite file (set to `/var/data/the_layman.db` on Render) |
| `ADMIN_USERNAME` | *(unset)* | Auto-creates admin on first boot |
| `ADMIN_PASSWORD` | *(unset)* | Password for auto-created admin |
| `FEED_SCHEDULE_HOUR` | `6` | UTC hour for daily feed generation |
| `LAYMAN_MODEL_TIMEOUT_S` | `600` | LLM request timeout |
| `LAYMAN_MODEL_TEMPERATURE` | `0` | Model temperature |
| `LAYMAN_ALLOW_GROUNDED_FALLBACK` | `0` | Set to `1` to allow fallback responses without a model |
