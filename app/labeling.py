"""LLM labeling + outlier refinement: 
Takes the raw output of app/clustering.py and asks the LLM to (a) 
name each cluster's vibe and (b) flag tracks that are a poor fit for 
their assigned cluster, suggesting a better one.
"""

from pydantic import BaseModel

from app.llm import generate_structured

SAMPLE_SIZE = 8  # tracks shown per cluster

_SYSTEM_PROMPT = (
    "You are a music curator assistant. You'll see several clusters of tracks "
    "that were grouped algorithmically by genre, mood/energy, and release era — "
    "each track is shown with its genre tags and mood tags separately (genre | "
    "mood). For each cluster:\n"
    "1. Give it a short, human-readable vibe label (2-5 words, e.g. 'Daniel "
    "Caesar R&B/Soul', 'Early BTS K-Pop', 'Bieber Dance-Pop Era', 'Late-Night "
    "House/Rave') based on what the sampled tracks have in common — lean on "
    "mood tags, not just genre, when a cluster's tracks share a genre but the "
    "real common thread is energy/setting (e.g. several 'house' tracks that are "
    "all tagged 'rave'/'high-energy' should be labeled around that, not just "
    "'House').\n"
    "2. Look across ALL clusters and flag any track that clearly fits another "
    "cluster's vibe better than the one it's currently in — this happens when a "
    "track's genre, mood, or artist doesn't match the rest of its cluster but "
    "matches another cluster well. Only flag clear, confident mismatches; most "
    "tracks should not be flagged. Never suggest moving a track to the cluster "
    "it's already in."
)


class _ClusterLabel(BaseModel):
    cluster_id: int
    label: str


class _Misfit(BaseModel):
    track_id: str
    current_cluster_id: int
    suggested_cluster_id: int
    reason: str


class _LabelingResponse(BaseModel):
    cluster_labels: list[_ClusterLabel]
    misfits: list[_Misfit]


def _format_track(track: dict) -> str:
    artists = ", ".join(a["name"] for a in track.get("artists", []) if a.get("name"))
    genres = ", ".join(track.get("genres", [])) or "no genre data"
    moods = ", ".join(track.get("moods", [])) or "no mood data"
    year = (track.get("release_date") or "")[:4]
    return f"  - [{track.get('id')}] \"{track.get('name')}\" by {artists} ({genres} | {moods}) [{year}]"


def _build_prompt(clusters: list[dict]) -> str:
    sections = []
    for cluster in clusters:
        tracks = cluster["tracks"]
        sample = tracks[:SAMPLE_SIZE]
        lines = [_format_track(t) for t in sample]
        if len(tracks) > len(sample):
            lines.append(f"  ... and {len(tracks) - len(sample)} more tracks in this cluster")
        sections.append(f"Cluster {cluster['cluster_id']} ({len(tracks)} tracks total):\n" + "\n".join(lines))
    return "\n\n".join(sections)


async def _call_llm(prompt: str) -> _LabelingResponse:
    return await generate_structured(_SYSTEM_PROMPT, prompt, _LabelingResponse)


async def label_and_refine_clusters(cluster_result: dict) -> dict:
    """Takes cluster_tracks()'s output, returns the same shape with a
    `label` added to each cluster and misfit tracks actually moved to
    their suggested cluster."""
    clusters = cluster_result["clusters"]
    if len(clusters) < 2:
        for cluster in clusters:
            cluster["label"] = "All Tracks"
        return cluster_result

    result = await _call_llm(_build_prompt(clusters))

    tracks_by_id: dict[str, dict] = {}
    cluster_by_id: dict[int, dict] = {c["cluster_id"]: c for c in clusters}
    for cluster in clusters:
        for track in cluster["tracks"]:
            if track.get("id"):
                tracks_by_id[track["id"]] = track

    moved: set[str] = set()
    for misfit in result.misfits:
        track = tracks_by_id.get(misfit.track_id)
        source = cluster_by_id.get(misfit.current_cluster_id)
        target = cluster_by_id.get(misfit.suggested_cluster_id)
        
        if not track or not source or not target or target is source or misfit.track_id in moved:
            continue
        if track not in source["tracks"]:
            continue
        source["tracks"].remove(track)
        target["tracks"].append(track)
        moved.add(misfit.track_id)

    for cluster in clusters:
        cluster["track_count"] = len(cluster["tracks"])

    labels_by_id = {label.cluster_id: label.label for label in result.cluster_labels}
    for cluster in clusters:
        cluster["label"] = labels_by_id.get(cluster["cluster_id"], f"Cluster {cluster['cluster_id']}")

    cluster_result["clusters"] = [c for c in clusters if c["track_count"] > 0]
    return cluster_result
