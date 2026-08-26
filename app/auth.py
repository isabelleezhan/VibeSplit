import secrets
import time
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])

SPOTIFY_AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SESSION_COOKIE_NAME = "session_id"

SCOPES = " ".join(
    [
        "user-library-read",
        "playlist-read-private",
        "playlist-modify-private",
    ]
)

# NOTE: Once Postgres is wired up, swap this for the `users` table and drop these
# process-global dicts
_pending_states: set[str] = set()
_sessions: dict[str, dict] = {}


def _store_session(token_data: dict) -> str:
    session_id = secrets.token_urlsafe(24)
    _sessions[session_id] = {
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": time.time() + token_data.get("expires_in", 3600),
    }
    return session_id


@router.get("/login")
def login() -> RedirectResponse:
    settings = get_settings()
    if not settings.spotify_client_id:
        raise HTTPException(
            status_code=500,
            detail="SPOTIFY_CLIENT_ID is not set — check your .env file.",
        )

    state = secrets.token_urlsafe(16)
    _pending_states.add(state)

    params = {
        "response_type": "code",
        "client_id": settings.spotify_client_id,
        "scope": SCOPES,
        "redirect_uri": settings.spotify_redirect_uri,
        "state": state,
    }
    return RedirectResponse(f"{SPOTIFY_AUTHORIZE_URL}?{urlencode(params)}")


@router.get("/callback")
async def callback(request: Request) -> JSONResponse:
    settings = get_settings()

    error = request.query_params.get("error")
    if error:
        raise HTTPException(status_code=400, detail=f"Spotify returned an error: {error}")

    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing 'code' or 'state' in callback.")
    if state not in _pending_states:
        raise HTTPException(status_code=400, detail="Unknown or reused 'state' — possible CSRF, please retry /auth/login.")
    _pending_states.discard(state)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            SPOTIFY_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.spotify_redirect_uri,
            },
            auth=(settings.spotify_client_id, settings.spotify_client_secret),
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Spotify token exchange failed: {response.status_code} {response.text}",
        )

    token_data = response.json()
    session_id = _store_session(token_data)

    resp = JSONResponse(
        {
            "message": "Spotify auth successful",
            "scope": token_data.get("scope"),
            "expires_in": token_data.get("expires_in"),
        }
    )
    # NOTE: secure=False because local dev runs over plain http on 127.0.0.1.
    # Set secure=True once this is served over https.
    resp.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
    )
    return resp


async def _refresh(refresh_token: str) -> dict:
    settings = get_settings()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            SPOTIFY_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            auth=(settings.spotify_client_id, settings.spotify_client_secret),
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Spotify token refresh failed: {response.status_code} {response.text}",
        )

    return response.json()


async def get_session_token(request: Request) -> str:
    """FastAPI dependency: resolve the caller's session cookie to a valid
    Spotify access token, transparently refreshing it if it's expired."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    session = _sessions.get(session_id) if session_id else None
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated — visit /auth/login first.")

    # 30s buffer so we don't hand out a token that expires mid-request.
    if session["expires_at"] <= time.time() + 30:
        if not session["refresh_token"]:
            raise HTTPException(status_code=401, detail="Session expired — visit /auth/login again.")
        refreshed = await _refresh(session["refresh_token"])
        session["access_token"] = refreshed["access_token"]
        session["expires_at"] = time.time() + refreshed.get("expires_in", 3600)
        # Spotify only sometimes rotates the refresh token.
        session["refresh_token"] = refreshed.get("refresh_token", session["refresh_token"])

    return session["access_token"]
