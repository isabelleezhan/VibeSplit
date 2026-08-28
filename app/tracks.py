from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_session_token
from app.genre_tagging import tag_artist_genres

router = APIRouter(prefix="/playlists", tags=["tracks"])

SPOTIFY_API_BASE = "https://api.spotify.com/v1"


async def _get_all_playlist_items(client: httpx.AsyncClient, playlist_id: str, access_token: str) -> list[dict]:
    """Page through a playlist's tracks. Spotify caps each page at 100 items
    and gives a `next` URL for the following page, or null when done."""
    items: list[dict] = []
    url = f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/items"
    params = {"limit": 100}

    while url:
        response = await client.get(url, headers={"Authorization": f"Bearer {access_token}"}, params=params)
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Spotify playlist items request failed: {response.status_code} {response.text}",
            )
        data = response.json()
        items.extend(data.get("items", []))
        url = data.get("next")
        params = None 

    return items


async def get_enriched_playlist_tracks(playlist_id: str, access_token: str) -> list[dict]:
    """Step 2 (track import) + step 3 (genre enrichment) of the build order,
    combined: every track in the playlist, with genre/vibe tags attached.

    NOTE: enrichment come from a single batched LLM call
    over the playlist's unique artist names (see app/genre_tagging.py).
    """
    async with httpx.AsyncClient() as client:
        items = await _get_all_playlist_items(client, playlist_id, access_token)

    tracks = []
    artist_names: set[str] = set()

    for item in items:
        track = item.get("item")
        if not track:
            continue  # e.g. a track that was removed from Spotify entirely.

        artists = [
            {"id": a.get("id"), "name": a.get("name")}
            for a in (track.get("artists") or [])
            if a
        ]
        for a in artists:
            if a["name"]:
                artist_names.add(a["name"])

        tracks.append(
            {
                "id": track.get("id"),
                "name": track.get("name"),
                "is_local": item.get("is_local", False),
                "popularity": track.get("popularity"),
                "release_date": (track.get("album") or {}).get("release_date"),
                "artists": artists,
            }
        )

    tags_by_artist_name = await tag_artist_genres(sorted(artist_names))

    for track in tracks:
        # Union of all this track's artists' tags.
        genres: set[str] = set()
        for artist in track["artists"]:
            genres.update(tags_by_artist_name.get(artist["name"], []))
        track["genres"] = sorted(genres)

    return tracks


@router.get("/{playlist_id}/tracks")
async def list_playlist_tracks(
    playlist_id: str,
    access_token: Annotated[str, Depends(get_session_token)],
) -> dict:
    tracks = await get_enriched_playlist_tracks(playlist_id, access_token)
    return {"total": len(tracks), "tracks": tracks}
