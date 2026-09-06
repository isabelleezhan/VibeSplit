import time
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_session_token
from app.db import get_db
from app.genre_tagging import tag_tracks
from app.models import TrackTagCache

router = APIRouter(prefix="/playlists", tags=["tracks"])

SPOTIFY_API_BASE = "https://api.spotify.com/v1"

# How long a cached tag entry stays valid before it's treated as a miss
# and re-tagged.
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days


async def _get_all_playlist_items(client: httpx.AsyncClient, playlist_id: str, access_token: str) -> list[dict]:
    """Page through a playlist's tracks. Spotify caps each page at 100 items
    and gives a `next` URL for the following page, or null when done."""
    items: list[dict] = []
    url = f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/items"
    params = {"limit": 100}

    # Keep fetching pages until Spotify stops giving a `next` URL.
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


async def _get_cached_tags(db: AsyncSession, track_ids: set[str]) -> dict[str, dict[str, list[str]]]:
    """Returns {track_id: {"genres": [...], "moods": [...]}} for whichever
    of the given ids have a not-yet-expired cache entry."""
    if not track_ids:
        return {}
    result = await db.execute(select(TrackTagCache).where(TrackTagCache.track_id.in_(track_ids)))
    now = time.time()
    return {
        row.track_id: {"genres": row.genres, "moods": row.moods}
        for row in result.scalars()
        if now - row.tagged_at < CACHE_TTL_SECONDS
    }


async def _save_tags_to_cache(db: AsyncSession, tags_by_track_id: dict[str, dict[str, list[str]]]) -> None:
    """Upserts freshly-tagged tracks into the cache. `tags_by_track_id`
    should already be filtered down to real, cacheable Spotify ids."""
    if not tags_by_track_id:
        return
    now = time.time()
    for track_id, tags in tags_by_track_id.items():
        cached = await db.get(TrackTagCache, track_id)
        if cached:
            cached.genres = tags["genres"]
            cached.moods = tags["moods"]
            cached.tagged_at = now
        else:
            db.add(TrackTagCache(track_id=track_id, genres=tags["genres"], moods=tags["moods"], tagged_at=now))
    await db.commit()


async def get_enriched_playlist_tracks(playlist_id: str, access_token: str, db: AsyncSession) -> list[dict]:
    """Every track in the playlist, with genre tags AND mood/energy
    tags attached (`genres` and `moods`).

    NOTE: enrichment comes from a batched LLM call over whichever tracks
    aren't already cached (see app/genre_tagging.py and the cache helpers
    above).
    """
    async with httpx.AsyncClient() as client:
        items = await _get_all_playlist_items(client, playlist_id, access_token)

    tracks = []
    for item in items:
        track = item.get("item")
        if not track:
            continue  # e.g. a track that was removed from Spotify entirely.

        artists = [
            {"id": a.get("id"), "name": a.get("name")}
            for a in (track.get("artists") or [])
            if a
        ]

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

    for i, track in enumerate(tracks):
        track["_tag_key"] = track["id"] or f"_local_{i}"
    cacheable_ids = {track["id"] for track in tracks if track["id"]}

    cached_tags = await _get_cached_tags(db, cacheable_ids)
    tracks_needing_tagging = [t for t in tracks if t["_tag_key"] not in cached_tags]

    freshly_tagged = await tag_tracks(tracks_needing_tagging)
    newly_cacheable = {key: tags for key, tags in freshly_tagged.items() if key in cacheable_ids}
    await _save_tags_to_cache(db, newly_cacheable)

    tags_by_key = {**cached_tags, **freshly_tagged}

    for track in tracks:
        tags = tags_by_key.get(track.pop("_tag_key"), {})
        track["genres"] = sorted(tags.get("genres", []))
        track["moods"] = sorted(tags.get("moods", []))

    return tracks


@router.get("/{playlist_id}/tracks")
async def list_playlist_tracks(
    playlist_id: str,
    access_token: Annotated[str, Depends(get_session_token)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    tracks = await get_enriched_playlist_tracks(playlist_id, access_token, db)
    return {"total": len(tracks), "tracks": tracks}
