"""Clustering engine: Groups tracks into "vibe" clusters using genre-overlap
(Jaccard) distance, blended with mood/energy-overlap distance and
normalized release-era distance.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

GENRE_WEIGHT = 0.5
MOOD_WEIGHT = 0.3
ERA_WEIGHT = 0.2

MIN_CLUSTERS = 2
MAX_CLUSTERS = 5
NEUTRAL_DISTANCE = 0.5  # used when a numeric feature is missing on either side

MIN_TRACKS_FOR_COHERENCE = 4

def _extract_year(release_date: str | None) -> float | None:
    if not release_date:
        return None
    try:
        return float(release_date[:4])
    except ValueError:
        return None


def _normalize(values: list[float | None]) -> list[float | None]:
    """Min-max normalize to [0, 1], preserving None for missing values."""
    present = [v for v in values if v is not None]
    if not present:
        return [None] * len(values)
    lo, hi = min(present), max(present)
    if hi == lo:
        return [0.0 if v is not None else None for v in values]
    return [None if v is None else (v - lo) / (hi - lo) for v in values]


def _numeric_distance(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return NEUTRAL_DISTANCE
    return abs(a - b)


def _tag_overlap_distance(tags_a: set[str], tags_b: set[str]) -> float | None:
    """Jaccard distance between two tag sets — 0 = identical, 1 = disjoint.
    None (not a number) when either side has no tags to compare"""
    if not tags_a or not tags_b:
        return None
    return 1 - len(tags_a & tags_b) / len(tags_a | tags_b)


def _pairwise_distance(a: dict, b: dict) -> float:
    era_distance = _numeric_distance(a["_year_norm"], b["_year_norm"])
    genre_distance = _tag_overlap_distance(set(a["genres"]), set(b["genres"]))
    mood_distance = _tag_overlap_distance(set(a.get("moods", [])), set(b.get("moods", [])))

    components = [(GENRE_WEIGHT, genre_distance), (MOOD_WEIGHT, mood_distance), (ERA_WEIGHT, era_distance)]
    available = [(weight, distance) for weight, distance in components if distance is not None]
    total_weight = sum(weight for weight, _ in available)
    return sum(weight * distance for weight, distance in available) / total_weight


def playlist_year_range(tracks: list[dict]) -> tuple[float, float] | None:
    years = [_extract_year(t.get("release_date")) for t in tracks]
    present = [y for y in years if y is not None]
    return (min(present), max(present)) if present else None


def compute_coherence(tracks: list[dict], year_range: tuple[float, float] | None = None) -> float | None:
    """Average pairwise similarity within a group of tracks (1 - average
    pairwise distance)
    """
    if len(tracks) < MIN_TRACKS_FOR_COHERENCE:
        return None

    if year_range is None:
        year_range = playlist_year_range(tracks)

    if year_range is None:
        year_norm = [None] * len(tracks)
    else:
        lo, hi = year_range
        denom = hi - lo if hi > lo else 1.0
        year_norm = [None if (y := _extract_year(t.get("release_date"))) is None else (y - lo) / denom for t in tracks]

    enriched = [{**track, "_year_norm": y} for track, y in zip(tracks, year_norm)]

    distances = [
        _pairwise_distance(enriched[i], enriched[j])
        for i in range(len(enriched))
        for j in range(i + 1, len(enriched))
    ]
    avg_distance = sum(distances) / len(distances)
    return round(1 - avg_distance, 4)


def _max_k(n: int) -> int:
    return max(MIN_CLUSTERS, min(MAX_CLUSTERS, n - 1))


def _single_cluster_result(tracks: list[dict]) -> dict:
    return {
        "k": 1,
        "silhouette_score": None,
        "clusters": [{"cluster_id": 0, "track_count": len(tracks), "tracks": tracks}],
    }


def cluster_tracks(tracks: list[dict]) -> dict:
    """Cluster tracks by vibe. Returns the chosen k, its silhouette score,
    and the resulting groups. Falls back to a single cluster when there
    aren't enough tracks to meaningfully split (fewer than MIN_CLUSTERS + 1)."""
    n = len(tracks)
    if n < MIN_CLUSTERS + 1:
        return _single_cluster_result(tracks)

    year_norm = _normalize([_extract_year(t.get("release_date")) for t in tracks])
    enriched = [{**track, "_year_norm": y} for track, y in zip(tracks, year_norm)]

    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _pairwise_distance(enriched[i], enriched[j])
            distance_matrix[i, j] = distance_matrix[j, i] = d

    best_k, best_score, best_labels = None, -1.0, None
    for k in range(MIN_CLUSTERS, _max_k(n) + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(distance_matrix)
        if len(set(labels)) < 2:
            continue  # silhouette needs at least 2 distinct clusters
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None:
        # Every track ended up equidistant/identical — degenerate case.
        return _single_cluster_result(tracks)

    clusters: dict[int, list[dict]] = {}
    for track, label in zip(tracks, best_labels):
        clusters.setdefault(int(label), []).append(track)

    return {
        "k": best_k,
        "silhouette_score": round(float(best_score), 4),
        "clusters": [
            {"cluster_id": cluster_id, "track_count": len(members), "tracks": members}
            for cluster_id, members in sorted(clusters.items())
        ],
    }
