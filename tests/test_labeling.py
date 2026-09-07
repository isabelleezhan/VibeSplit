"""Unit tests for app/labeling.py"""

from unittest.mock import AsyncMock, patch

import pytest

from app.labeling import _ClusterLabel, _LabelingResponse, _Misfit, label_and_refine_clusters


def make_cluster(cluster_id: int, track_ids: list[str]) -> dict:
    tracks = [{"id": tid, "name": tid, "artists": [], "genres": [], "moods": [], "release_date": None} for tid in track_ids]
    return {"cluster_id": cluster_id, "track_count": len(tracks), "tracks": tracks}


def cluster_result(clusters: list[dict]) -> dict:
    return {"k": len(clusters), "silhouette_score": 0.5, "clusters": clusters}


@pytest.fixture
def mock_llm():
    """Patches the actual network call so tests control exactly what the
    'LLM' returns, without touching the real API."""
    with patch("app.labeling.generate_structured", new=AsyncMock()) as mocked:
        yield mocked


class TestLabelAndRefineClusters:
    async def test_single_cluster_skips_the_llm_call(self, mock_llm):
        result = await label_and_refine_clusters(cluster_result([make_cluster(0, ["t1", "t2"])]))
        assert result["clusters"][0]["label"] == "All Tracks"
        mock_llm.assert_not_called()

    async def test_labels_get_applied_to_the_right_clusters(self, mock_llm):
        clusters = [make_cluster(0, ["t1", "t2"]), make_cluster(1, ["t3", "t4"])]
        mock_llm.return_value = _LabelingResponse(
            cluster_labels=[
                _ClusterLabel(cluster_id=0, label="Late-Night House"),
                _ClusterLabel(cluster_id=1, label="90s Rock"),
            ],
            misfits=[],
        )
        result = await label_and_refine_clusters(cluster_result(clusters))
        labels_by_id = {c["cluster_id"]: c["label"] for c in result["clusters"]}
        assert labels_by_id == {0: "Late-Night House", 1: "90s Rock"}

    async def test_missing_label_falls_back_to_generic_name(self, mock_llm):
        clusters = [make_cluster(0, ["t1"]), make_cluster(1, ["t2"])]
        mock_llm.return_value = _LabelingResponse(
            cluster_labels=[_ClusterLabel(cluster_id=0, label="Only This One")],
            misfits=[],
        )
        result = await label_and_refine_clusters(cluster_result(clusters))
        labels_by_id = {c["cluster_id"]: c["label"] for c in result["clusters"]}
        assert labels_by_id[0] == "Only This One"
        assert labels_by_id[1] == "Cluster 1"

    async def test_misfit_track_actually_moves_clusters(self, mock_llm):
        clusters = [make_cluster(0, ["t1", "t2"]), make_cluster(1, ["t3", "t4"])]
        mock_llm.return_value = _LabelingResponse(
            cluster_labels=[_ClusterLabel(cluster_id=0, label="A"), _ClusterLabel(cluster_id=1, label="B")],
            misfits=[_Misfit(track_id="t2", current_cluster_id=0, suggested_cluster_id=1, reason="better fit")],
        )
        result = await label_and_refine_clusters(cluster_result(clusters))
        by_id = {c["cluster_id"]: c for c in result["clusters"]}
        moved_ids = {t["id"] for t in by_id[1]["tracks"]}
        remaining_ids = {t["id"] for t in by_id[0]["tracks"]}
        assert "t2" in moved_ids
        assert "t2" not in remaining_ids
        assert by_id[0]["track_count"] == 1
        assert by_id[1]["track_count"] == 3

    async def test_hallucinated_track_id_is_ignored(self, mock_llm):
        # The LLM naming a track id that doesn't exist anywhere shouldn't
        # crash the request.
        clusters = [make_cluster(0, ["t1"]), make_cluster(1, ["t2"])]
        mock_llm.return_value = _LabelingResponse(
            cluster_labels=[_ClusterLabel(cluster_id=0, label="A"), _ClusterLabel(cluster_id=1, label="B")],
            misfits=[_Misfit(track_id="does-not-exist", current_cluster_id=0, suggested_cluster_id=1, reason="?")],
        )
        result = await label_and_refine_clusters(cluster_result(clusters))
        sizes = {c["cluster_id"]: c["track_count"] for c in result["clusters"]}
        assert sizes == {0: 1, 1: 1}

    async def test_suggesting_the_same_cluster_is_a_noop(self, mock_llm):
        clusters = [make_cluster(0, ["t1", "t2"]), make_cluster(1, ["t3"])]
        mock_llm.return_value = _LabelingResponse(
            cluster_labels=[_ClusterLabel(cluster_id=0, label="A"), _ClusterLabel(cluster_id=1, label="B")],
            misfits=[_Misfit(track_id="t1", current_cluster_id=0, suggested_cluster_id=0, reason="?")],
        )
        result = await label_and_refine_clusters(cluster_result(clusters))
        by_id = {c["cluster_id"]: c["track_count"] for c in result["clusters"]}
        assert by_id == {0: 2, 1: 1}

    async def test_cluster_emptied_by_misfits_is_dropped(self, mock_llm):
        clusters = [make_cluster(0, ["t1"]), make_cluster(1, ["t2", "t3"])]
        mock_llm.return_value = _LabelingResponse(
            cluster_labels=[_ClusterLabel(cluster_id=0, label="A"), _ClusterLabel(cluster_id=1, label="B")],
            misfits=[_Misfit(track_id="t1", current_cluster_id=0, suggested_cluster_id=1, reason="only track moved out")],
        )
        result = await label_and_refine_clusters(cluster_result(clusters))
        remaining_ids = {c["cluster_id"] for c in result["clusters"]}
        assert remaining_ids == {1}

    async def test_double_move_of_the_same_track_is_prevented(self, mock_llm):
        clusters = [make_cluster(0, ["t1"]), make_cluster(1, []), make_cluster(2, [])]
        mock_llm.return_value = _LabelingResponse(
            cluster_labels=[_ClusterLabel(cluster_id=i, label=str(i)) for i in range(3)],
            misfits=[
                _Misfit(track_id="t1", current_cluster_id=0, suggested_cluster_id=1, reason="a"),
                _Misfit(track_id="t1", current_cluster_id=0, suggested_cluster_id=2, reason="b"),
            ],
        )
        result = await label_and_refine_clusters(cluster_result(clusters))
        sizes = {c["cluster_id"]: c["track_count"] for c in result["clusters"]}
        assert sizes.get(1) == 1
        assert sizes.get(2, 0) == 0
