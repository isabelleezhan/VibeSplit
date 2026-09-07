"""Unit tests for app/clustering.py"""

import pytest

from app.clustering import (
    MIN_TRACKS_FOR_COHERENCE,
    _extract_year,
    _normalize,
    _numeric_distance,
    _pairwise_distance,
    _tag_overlap_distance,
    cluster_tracks,
    compute_coherence,
)


def make_track(genre: str, year: int, mood: str | None = None) -> dict:
    return {"genres": [genre], "moods": [mood] if mood else [], "_year_norm": 0.5}


class TestExtractYear:
    def test_valid_date(self):
        assert _extract_year("2020-05-14") == 2020.0

    def test_none(self):
        assert _extract_year(None) is None

    def test_empty_string(self):
        assert _extract_year("") is None

    def test_malformed(self):
        assert _extract_year("not-a-date") is None


class TestNormalize:
    def test_spread(self):
        assert _normalize([2000.0, 2010.0, 2020.0]) == [0.0, 0.5, 1.0]

    def test_all_same_value(self):
        assert _normalize([2020.0, 2020.0]) == [0.0, 0.0]

    def test_all_none(self):
        assert _normalize([None, None]) == [None, None]

    def test_mixed_none_preserved(self):
        result = _normalize([2000.0, None, 2020.0])
        assert result[1] is None
        assert result[0] == 0.0
        assert result[2] == 1.0


class TestNumericDistance:
    def test_both_present(self):
        assert _numeric_distance(0.2, 0.8) == pytest.approx(0.6)

    def test_either_missing_falls_back_to_neutral(self):
        from app.clustering import NEUTRAL_DISTANCE

        assert _numeric_distance(None, 0.5) == NEUTRAL_DISTANCE
        assert _numeric_distance(0.5, None) == NEUTRAL_DISTANCE
        assert _numeric_distance(None, None) == NEUTRAL_DISTANCE


class TestTagOverlapDistance:
    def test_identical_sets(self):
        assert _tag_overlap_distance({"pop", "r&b"}, {"pop", "r&b"}) == 0.0

    def test_disjoint_sets(self):
        assert _tag_overlap_distance({"pop"}, {"metal"}) == 1.0

    def test_partial_overlap(self):
        # 1 shared / 3 total union -> distance 1 - 1/3
        assert _tag_overlap_distance({"pop", "r&b"}, {"pop", "hip hop"}) == pytest.approx(2 / 3)

    def test_either_side_empty_is_none_not_maximal_distance(self):
        assert _tag_overlap_distance(set(), {"pop"}) is None
        assert _tag_overlap_distance({"pop"}, set()) is None
        assert _tag_overlap_distance(set(), set()) is None


class TestPairwiseDistance:
    def test_identical_genre_mood_era(self):
        a = make_track("house", 2020, "rave")
        b = make_track("house", 2020, "rave")
        assert _pairwise_distance(a, b) == 0.0

    def test_same_genre_opposite_mood_is_not_identical(self):
        chill = make_track("house", 2020, "chill")
        rave = make_track("house", 2020, "rave")
        assert _pairwise_distance(chill, rave) > 0

    def test_missing_mood_data_is_dropped_not_penalized(self):
        with_mood = make_track("house", 2020, "rave")
        without_mood = make_track("house", 2020, mood=None)
        assert _pairwise_distance(with_mood, without_mood) == 0.0

    def test_genre_dominates_over_mood(self):
        from app.clustering import ERA_WEIGHT, GENRE_WEIGHT, MOOD_WEIGHT

        same_genre_diff_mood = _pairwise_distance(make_track("house", 2020, "chill"), make_track("house", 2020, "rave"))
        same_mood_diff_genre = _pairwise_distance(make_track("house", 2020, "chill"), make_track("jazz", 2020, "chill"))
        assert GENRE_WEIGHT > MOOD_WEIGHT  # sanity-check the config itself
        assert same_mood_diff_genre > same_genre_diff_mood


class TestComputeCoherence:
    def test_below_minimum_returns_none(self):
        tracks = [make_track("pop", 2020) for _ in range(MIN_TRACKS_FOR_COHERENCE - 1)]
        assert compute_coherence(tracks) is None

    def test_identical_tracks_are_maximally_coherent(self):
        tracks = [{"genres": ["pop"], "moods": ["chill"], "release_date": "2020-01-01"} for _ in range(5)]
        assert compute_coherence(tracks) == 1.0

    def test_scattered_tracks_score_lower_than_tight_ones(self):
        tight = [{"genres": ["pop"], "moods": ["chill"], "release_date": "2020-01-01"} for _ in range(5)]
        scattered = [
            {"genres": [g], "moods": [m], "release_date": d}
            for g, m, d in [
                ("pop", "chill", "2020-01-01"),
                ("metal", "aggressive", "1995-01-01"),
                ("jazz", "romantic", "1960-01-01"),
                ("house", "rave", "2015-01-01"),
                ("folk", "melancholy", "1975-01-01"),
            ]
        ]
        assert compute_coherence(tight) > compute_coherence(scattered)


class TestClusterTracks:
    def _tracks(self, n: int, genre: str, year: int) -> list[dict]:
        return [
            {"id": f"{genre}{i}", "genres": [genre], "moods": [], "release_date": f"{year}-01-01"}
            for i in range(n)
        ]

    def test_too_few_tracks_falls_back_to_single_cluster(self):
        # MIN_CLUSTERS is 2, so fewer than 3 tracks can't be meaningfully split.
        result = cluster_tracks(self._tracks(2, "pop", 2020))
        assert result["k"] == 1
        assert len(result["clusters"]) == 1

    def test_three_distinct_groups_split_into_three_clusters(self):
        tracks = self._tracks(8, "pop", 2020) + self._tracks(8, "metal", 1995) + self._tracks(8, "jazz", 1960)
        result = cluster_tracks(tracks)
        assert result["k"] == 3
        sizes = sorted(c["track_count"] for c in result["clusters"])
        assert sizes == [8, 8, 8]

    def test_small_clusters_are_allowed(self):
        tracks = self._tracks(4, "pop", 2020) + self._tracks(4, "metal", 1995) + self._tracks(4, "jazz", 1960)
        result = cluster_tracks(tracks)
        assert result["k"] == 3
        sizes = sorted(c["track_count"] for c in result["clusters"])
        assert sizes == [4, 4, 4]

    def test_every_track_is_accounted_for(self):
        tracks = self._tracks(5, "pop", 2020) + self._tracks(5, "metal", 1995)
        result = cluster_tracks(tracks)
        total = sum(c["track_count"] for c in result["clusters"])
        assert total == len(tracks)

    def test_fully_identical_tracks_still_partition_cleanly(self):
        tracks = [{"id": f"t{i}", "genres": ["pop"], "moods": [], "release_date": "2020-01-01"} for i in range(6)]
        result = cluster_tracks(tracks)
        assert sum(c["track_count"] for c in result["clusters"]) == len(tracks)

