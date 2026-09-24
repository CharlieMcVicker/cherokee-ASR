# -*- coding: utf-8 -*-
"""
Tests for PhoneticManifoldAnalyzer and ManifoldVisualizer.
"""

from pathlib import Path
import numpy as np
import pytest

from digohwelisgi.evaluation.manifold import PhoneticManifoldAnalyzer
from digohwelisgi.evaluation.visualizer import ManifoldVisualizer


class TestPhoneticManifoldAnalyzer:
    """Unit tests for PhoneticManifoldAnalyzer."""

    @pytest.fixture
    def simple_confusion(self):
        """
        Create a 4x4 mock conditional probability matrix.
        Labels: ['a', 'b', 'c', 'd']
        Components with threshold 0.05:
          - Cluster 1: ('a', 'b') via P(b|a)=0.20, P(a|b)=0.10 -> A[0, 1] = 0.15
          - Cluster 2: ('c', 'd') via P(d|c)=0.12, P(c|d)=0.08 -> A[2, 3] = 0.10
          - Cross-cluster entries < 0.05
        """
        labels = ["a", "b", "c", "d"]
        cond_probs = np.array(
            [
                [0.78, 0.20, 0.01, 0.01],
                [0.10, 0.88, 0.01, 0.01],
                [0.01, 0.01, 0.86, 0.12],
                [0.01, 0.01, 0.08, 0.90],
            ],
            dtype=np.float64,
        )
        return cond_probs, labels

    def test_init_validation(self, simple_confusion):
        cond_probs, labels = simple_confusion
        analyzer = PhoneticManifoldAnalyzer(cond_probs, labels)
        assert analyzer.n_labels == 4
        assert analyzer.labels == ["a", "b", "c", "d"]

        # Test non-square matrix raises ValueError
        with pytest.raises(ValueError, match="2D square matrix"):
            PhoneticManifoldAnalyzer(np.ones((4, 3)), labels)

        # Test 1D array raises ValueError
        with pytest.raises(ValueError, match="2D square matrix"):
            PhoneticManifoldAnalyzer(np.ones((4,)), labels)

        # Test mismatched labels count raises ValueError
        with pytest.raises(ValueError, match="does not match number of labels"):
            PhoneticManifoldAnalyzer(cond_probs, ["a", "b", "c"])

    def test_get_adjacency_matrix(self, simple_confusion):
        cond_probs, labels = simple_confusion
        analyzer = PhoneticManifoldAnalyzer(cond_probs, labels)

        # Default diagonal=0.0
        A = analyzer.get_adjacency_matrix(diagonal=0.0)
        assert A.shape == (4, 4)
        assert np.allclose(A, A.T)
        assert np.allclose(np.diag(A), 0.0)
        assert np.isclose(A[0, 1], 0.15)
        assert np.isclose(A[1, 0], 0.15)
        assert np.isclose(A[2, 3], 0.10)
        assert np.isclose(A[3, 2], 0.10)

        # Custom diagonal=1.0
        A_diag1 = analyzer.get_adjacency_matrix(diagonal=1.0)
        assert np.allclose(np.diag(A_diag1), 1.0)

        # Preserve diagonal
        A_none = analyzer.get_adjacency_matrix(diagonal=None)
        assert np.isclose(A_none[0, 0], 0.78)

    def test_get_connected_components(self, simple_confusion):
        cond_probs, labels = simple_confusion
        analyzer = PhoneticManifoldAnalyzer(cond_probs, labels)

        # Threshold 0.05 partitions into [0, 1] and [2, 3]
        comps = analyzer.get_connected_components(threshold=0.05)
        assert len(comps) == 2
        assert [0, 1] in comps
        assert [2, 3] in comps

        # Threshold 0.20 isolates all 4 nodes
        comps_high = analyzer.get_connected_components(threshold=0.20)
        assert len(comps_high) == 4
        assert sorted(comps_high) == [[0], [1], [2], [3]]

        # Threshold 0.00 connects all nodes into 1 component
        comps_low = analyzer.get_connected_components(threshold=0.00)
        assert len(comps_low) == 1
        assert comps_low[0] == [0, 1, 2, 3]

    def test_get_component_labels(self, simple_confusion):
        cond_probs, labels = simple_confusion
        analyzer = PhoneticManifoldAnalyzer(cond_probs, labels)

        comp_labels = analyzer.get_component_labels(threshold=0.05)
        assert len(comp_labels) == 2
        assert ["a", "b"] in comp_labels
        assert ["c", "d"] in comp_labels

    def test_get_block_diagonal_reordering(self):
        # 5 labels with clusters {0, 4}, {2}, {1, 3}
        labels = ["t", "k", "s", "g", "d"]
        cond_probs = np.zeros((5, 5))
        # cluster 0-4 ('t'-'d')
        cond_probs[0, 4] = 0.2
        cond_probs[4, 0] = 0.2
        # cluster 1-3 ('k'-'g')
        cond_probs[1, 3] = 0.15
        cond_probs[3, 1] = 0.15
        # 2 ('s') isolated

        analyzer = PhoneticManifoldAnalyzer(cond_probs, labels)
        perm_idx, perm_lbl = analyzer.get_block_diagonal_reordering(threshold=0.05)

        assert len(perm_idx) == 5
        assert set(perm_idx) == {0, 1, 2, 3, 4}
        assert perm_lbl == [labels[i] for i in perm_idx]

        # Verify contiguous grouping:
        # Either [0, 4] appears contiguously or [1, 3] appears contiguously
        idx_0 = perm_idx.index(0)
        idx_4 = perm_idx.index(4)
        assert abs(idx_0 - idx_4) == 1

        idx_1 = perm_idx.index(1)
        idx_3 = perm_idx.index(3)
        assert abs(idx_1 - idx_3) == 1

    def test_get_component_slices_and_reorder_matrix(self):
        labels = ["a", "b", "c", "d", "e"]
        cond_probs = np.zeros((5, 5))
        cond_probs[0, 1] = 0.3
        cond_probs[1, 0] = 0.3
        cond_probs[2, 3] = 0.3
        cond_probs[3, 2] = 0.3

        analyzer = PhoneticManifoldAnalyzer(cond_probs, labels)
        slices = analyzer.get_component_slices(threshold=0.05, min_size=2)
        assert slices == [(0, 2), (2, 4)]

        all_slices = analyzer.get_component_slices(threshold=0.05, min_size=1)
        assert all_slices == [(0, 2), (2, 4), (4, 5)]

        perm_idx, _ = analyzer.get_block_diagonal_reordering(threshold=0.05)
        reordered = analyzer.reorder_matrix(cond_probs, perm_idx)
        assert reordered.shape == (5, 5)

        with pytest.raises(ValueError, match="does not match permutation length"):
            analyzer.reorder_matrix(np.zeros((4, 4)), perm_idx)

    def test_compute_manifold_stability(self):
        labels = ["a", "b", "c"]
        snr_levels = [30.0, 10.0, 0.0]

        # SNR 30: almost diagonal (all isolated)
        m_30 = np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]])
        # SNR 10: a-b confusion (2 components: [a, b], [c])
        m_10 = np.array([[0.80, 0.18, 0.02], [0.18, 0.80, 0.02], [0.02, 0.02, 0.96]])
        # SNR 0: a-b-c chain confusion (1 component: [a, b, c], path a-b-c has diameter 2)
        m_0 = np.array([[0.60, 0.35, 0.05], [0.35, 0.30, 0.35], [0.05, 0.35, 0.60]])

        analyzer = PhoneticManifoldAnalyzer(m_30, labels)
        res = analyzer.compute_manifold_stability(
            snr_levels=snr_levels,
            cond_probs_by_snr=[m_30, m_10, m_0],
            threshold=0.10,
        )

        assert res["snr_levels"] == [30.0, 10.0, 0.0]
        assert res["num_components"] == [3, 2, 1]
        assert res["component_counts"] == [3, 2, 1]
        assert res["max_diameters"] == [0, 1, 2]
        assert res["threshold"] == 0.10

        # Test classmethod compute_stability
        res_static = PhoneticManifoldAnalyzer.compute_stability(
            snr_levels=snr_levels,
            cond_probs_by_snr=[m_30, m_10, m_0],
            threshold=0.10,
            labels=labels,
        )
        assert res_static["num_components"] == [3, 2, 1]

        # Mismatched length error
        with pytest.raises(ValueError, match="does not match"):
            analyzer.compute_manifold_stability([30.0], [m_30, m_10])


class TestManifoldVisualizer:
    """Unit tests for ManifoldVisualizer."""

    @pytest.fixture
    def mock_matrices(self):
        labels = ["a", "b", "c", "d"]
        confusion = np.array(
            [
                [0.80, 0.15, 0.03, 0.02],
                [0.12, 0.82, 0.04, 0.02],
                [0.02, 0.03, 0.75, 0.20],
                [0.01, 0.02, 0.18, 0.79],
            ],
            dtype=np.float64,
        )
        cost = np.array(
            [
                [0.00, 0.25, 0.90, 0.95],
                [0.30, 0.00, 0.88, 0.92],
                [0.92, 0.85, 0.00, 0.20],
                [0.96, 0.90, 0.22, 0.00],
            ],
            dtype=np.float64,
        )
        return confusion, cost, labels

    def test_plot_side_by_side(self, tmp_path, mock_matrices):
        confusion, cost, labels = mock_matrices
        save_file = tmp_path / "heatmaps" / "side_by_side.png"

        slices = [(0, 2), (2, 4)]
        out_path = ManifoldVisualizer.plot_side_by_side(
            confusion_matrix=confusion,
            cost_matrix=cost,
            labels=labels,
            save_path=save_file,
            component_slices=slices,
            title="Test Side-by-Side Heatmaps",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert out_path == save_file

        # Test validation error on shape mismatch
        with pytest.raises(ValueError, match="confusion_matrix shape"):
            ManifoldVisualizer.plot_side_by_side(
                confusion_matrix=np.ones((3, 3)),
                cost_matrix=cost,
                labels=labels,
                save_path=tmp_path / "bad.png",
            )

    def test_plot_snr_drift(self, tmp_path):
        snr_levels = [30.0, 20.0, 10.0, 0.0]
        component_counts = [10, 8, 4, 1]
        cer_scores = [0.02, 0.05, 0.15, 0.45]

        # 1. With CER scores
        save_file_dual = tmp_path / "snr_drift_dual.png"
        out1 = ManifoldVisualizer.plot_snr_drift(
            snr_levels=snr_levels,
            component_counts=component_counts,
            cer_scores=cer_scores,
            save_path=save_file_dual,
            title="SNR Drift Test",
        )
        assert out1.exists()
        assert out1.stat().st_size > 0

        # 2. Without CER scores
        save_file_single = tmp_path / "snr_drift_single.png"
        out2 = ManifoldVisualizer.plot_snr_drift(
            snr_levels=snr_levels,
            component_counts=component_counts,
            save_path=save_file_single,
        )
        assert out2.exists()
        assert out2.stat().st_size > 0

        # Validation errors
        with pytest.raises(ValueError, match="does not match"):
            ManifoldVisualizer.plot_snr_drift([30.0, 20.0], [10])

        with pytest.raises(ValueError, match="does not match"):
            ManifoldVisualizer.plot_snr_drift(
                snr_levels, component_counts, cer_scores=[0.1]
            )

    def test_plot_3d_confusion_mesh(self, tmp_path, mock_matrices):
        confusion, _, labels = mock_matrices
        save_file = tmp_path / "mesh_3d.png"

        out_path = ManifoldVisualizer.plot_3d_confusion_mesh(
            confusion_matrix=confusion,
            labels=labels,
            save_path=save_file,
            title="3D Mesh Test",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Test large label set (> 20 labels) for subsampled ticks
        large_n = 30
        large_conf = np.eye(large_n)
        large_labels = [f"tok_{i}" for i in range(large_n)]
        save_large = tmp_path / "mesh_3d_large.png"

        out_large = ManifoldVisualizer.plot_3d_confusion_mesh(
            confusion_matrix=large_conf,
            labels=large_labels,
            save_path=save_large,
        )
        assert out_large.exists()
        assert out_large.stat().st_size > 0

        # Validation error
        with pytest.raises(ValueError, match="confusion_matrix shape"):
            ManifoldVisualizer.plot_3d_confusion_mesh(
                confusion_matrix=np.ones((2, 2)),
                labels=labels,
                save_path=tmp_path / "bad_3d.png",
            )

    def test_plot_regime_comparison(self, tmp_path, mock_matrices):
        confusion, cost, labels = mock_matrices
        save_file = tmp_path / "comparison.png"

        regimes = [
            {
                "name": "Low Noise",
                "confusion_matrix": confusion,
                "cost_matrix": cost,
                "component_slices": [(0, 2)],
            },
            {
                "name": "Mid Noise",
                "confusion_matrix": confusion,
                "cost_matrix": cost,
                "component_slices": None,
            },
            {
                "name": "High Noise",
                "confusion_matrix": confusion,
                "cost_matrix": cost,
            },
        ]

        out_path = ManifoldVisualizer.plot_regime_comparison(
            regimes=regimes,
            labels=labels,
            save_path=save_file,
            title="Comparison Test",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Test empty regimes error
        with pytest.raises(ValueError, match="regimes list must not be empty"):
            ManifoldVisualizer.plot_regime_comparison(
                regimes=[],
                labels=labels,
                save_path=tmp_path / "empty.png",
            )
