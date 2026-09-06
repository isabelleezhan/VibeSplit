from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_session_token
from app.clustering import cluster_tracks, compute_coherence, playlist_year_range
from app.db import get_db
from app.labeling import label_and_refine_clusters
from app.tracks import get_enriched_playlist_tracks

router = APIRouter(prefix="/playlists", tags=["clusters"])


@router.post("/{playlist_id}/clusters")
async def get_playlist_clusters(
    playlist_id: str,
    access_token: Annotated[str, Depends(get_session_token)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Import tracks, enrich with genres, cluster by vibe, then have
    the LLM name each cluster and reassign misfit
    tracks to a better-fitting cluster."""
    tracks = await get_enriched_playlist_tracks(playlist_id, access_token, db)

    # Fixed year basis so before/after scores are measured on the same scale
    year_range = playlist_year_range(tracks)
    original_coherence = compute_coherence(tracks, year_range=year_range)

    clustered = cluster_tracks(tracks)
    result = await label_and_refine_clusters(clustered)

    scored_coherences = []
    for cluster in result["clusters"]:
        coherence = compute_coherence(cluster["tracks"], year_range=year_range)
        cluster["coherence"] = coherence
        if coherence is not None:
            scored_coherences.append(coherence)

    result["original_coherence"] = original_coherence
    result["avg_new_coherence"] = (
        round(sum(scored_coherences) / len(scored_coherences), 4) if scored_coherences else None
    )
    return result
