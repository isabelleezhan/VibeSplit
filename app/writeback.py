from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import get_current_user
from app.models import User

"""
Two Spotify API calls, once per cluster:
    1. Create a playlist - POST /v1/me/playlists w/ JSON body like
        {"name": "...", "public": false}. Spotify hands back the new
        playlist's id. 
    2. Add tracks to it - POST /v1/playlists/{new_playlist_id}/tracks w/
        {"uris": ["spotify:track:{id}", ...]}.
"""

router = APIRouter(prefix="/playlists", tags=["writeback"])

SPOTIFY_API_BASE = "https://api.spotify.com/v1"
ADD_TRACKS_CHUNK_SIZE = 100  # Spotify's per-request cap


class ClusterToWrite(BaseModel):
    label: str
    tracks: list[dict]  # each needs at least "id" and "is_local"


class ConfirmSplitRequest(BaseModel):
    clusters: list[ClusterToWrite]


async def _create_playlist(client: httpx.AsyncClient, access_token: str, name: str) -> str:
    response = await client.post(
        f"{SPOTIFY_API_BASE}/me/playlists",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "name": name,
            "public": False,
            "description": "Created by VibeSplit",
        },
    )
    if response.status_code not in (200, 201):
        raise HTTPException(
            status_code=502,
            detail=f"Spotify create-playlist failed: {response.status_code} {response.text}",
        )
    return response.json()["id"]

async def _add_tracks(client: httpx.AsyncClient, access_token: str, playlist_id: str, uris: list[str]) -> None:
    i = 0
    while i < len(uris):
        chunk = uris[i: i + ADD_TRACKS_CHUNK_SIZE]
        response = await client.post(
            f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/items",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "uris": chunk
            },
        )
        i += ADD_TRACKS_CHUNK_SIZE
        if response.status_code not in (200, 201):
            raise HTTPException(
                status_code=502,
                detail=f"Spotify add_tracks failed: {response.status_code} {response.text}",
            )
    return
        
    

@router.post("/{playlist_id}/confirm-split")
async def confirm_split(
    playlist_id: str,
    body: ConfirmSplitRequest,
    user: Annotated[User, Depends(get_current_user)],  # gives both user.access_token and user.spotify_id
) -> dict:
    created = []
    async with httpx.AsyncClient() as client:
        for cluster in body.clusters:
            uris = [
                f"spotify:track:{t['id']}"  # build URI string from curr item t
                for t in cluster.tracks
                if t.get("id") and not t.get("is_local")
            ]
            if not uris:
                continue

            new_playlist_id = await _create_playlist(client, user.access_token, cluster.label)
            await _add_tracks(client, user.access_token, new_playlist_id, uris)
            created.append({"playlist_id": new_playlist_id, "name": cluster.label, "track_count": len(uris)})

    return {"created": created}
                
