# VibeSplit

Connect Spotify, pick a big/messy playlist, and split it into several smaller,
coherent "vibe" playlists — using genre/metadata-based clustering plus an LLM
to interpret and label the resulting groups.

## Stack

- **Backend:** FastAPI (Python)
- **Database:** Postgres
- **Cache / job queue backing:** Redis
- **Background jobs:** Celery or RQ
- **Frontend:** React
- **Auth:** Spotify OAuth (authorization code flow)
- **AI layer:** LLM API for cluster labeling + outlier detection (clustering
  itself is metadata-based, not embedding-based)

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment file and fill in real values (never commit `.env`):

```bash
cp .env.example .env
```

Run the dev server:

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/` for a hello-world check, or
`http://127.0.0.1:8000/health` for a health check.

## Project status

Early scaffold — FastAPI skeleton is up and running. Spotify OAuth is the
next milestone; see the project notes for the full build order.
