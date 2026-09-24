# -*- coding: utf-8 -*-
"""
transcription.evaluation.manifold

Phonetic manifold analysis via symmetrized confusion adjacency,
connected component extraction, and block-diagonal canonical reordering.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class PhoneticManifoldAnalyzer:
    """
    Analyzes phonetic confusion topology using graph-theoretic manifold projections.

    Constructs symmetrized adjacency matrices, partitions confusable phonemes into
    connected components across confidence thresholds, and computes canonical
    block-diagonal permutations for confusion basin visualization.
    """

    def __init__(self, cond_probs: np.ndarray, labels: list[str]) -> None:
        """
        Initialize PhoneticManifoldAnalyzer.

        Args:
            cond_probs: 2D square matrix of conditional probabilities P(hyp|ref)
                where shape is (N, N).
            labels: List of N character/token labels corresponding to matrix rows/columns.

        Raises:
            ValueError: If cond_probs is not a 2D square matrix or shape does not match len(labels).
        """
        arr = np.asarray(cond_probs, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"cond_probs must be a 2D square matrix, got shape {arr.shape}"
            )
        if arr.shape[0] != len(labels):
            raise ValueError(
                f"cond_probs shape {arr.shape} does not match number of labels ({len(labels)})"
            )

        self.cond_probs = arr
        self.labels = [str(lbl) for lbl in labels]
        self.n_labels = len(labels)

    def get_adjacency_matrix(self, diagonal: Optional[float] = 0.0) -> np.ndarray:
        """
        Compute symmetrized adjacency matrix from conditional confusion probabilities.

        A[i, j] = 0.5 * (P(j|i) + P(i|j))

        Args:
            diagonal: Optional scalar to overwrite diagonal elements (default: 0.0).
                If None, original diagonal is preserved.

        Returns:
            np.ndarray: Symmetrized (N, N) adjacency matrix.
        """
        A = 0.5 * (self.cond_probs + self.cond_probs.T)
        if diagonal is not None:
            np.fill_diagonal(A, float(diagonal))
        return A

    def get_connected_components(
        self, threshold: float = 0.05, sort_by_size: bool = True
    ) -> list[list[int]]:
        """
        Extract connected components of phonemes across confusion threshold tau.

        An undirected edge exists between distinct nodes i and j if A[i, j] > threshold.
        Connected components are discovered via breadth-first search.

        Args:
            threshold: Probability threshold tau for establishing an adjacency edge (default: 0.05).
            sort_by_size: If True, sort components in descending order of size, breaking
                ties by minimum index (default: True). If False, order by discovery.

        Returns:
            list[list[int]]: List of connected components, each containing node indices.
        """
        A = self.get_adjacency_matrix(diagonal=0.0)
        n = self.n_labels
        visited = set()
        components: list[list[int]] = []

        for i in range(n):
            if i not in visited:
                comp: list[int] = []
                queue = [i]
                visited.add(i)

                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)

                    # Find all unvisited neighbors with A[curr, j] > threshold
                    for j in range(n):
                        if j not in visited and j != curr and A[curr, j] > threshold:
                            visited.add(j)
                            queue.append(j)

                comp.sort()
                components.append(comp)

        if sort_by_size:
            components.sort(key=lambda c: (-len(c), c[0]))

        return components

    def get_component_labels(
        self, threshold: float = 0.05, sort_by_size: bool = True
    ) -> list[list[str]]:
        """
        Extract connected components with string label names.

        Args:
            threshold: Probability threshold tau.
            sort_by_size: If True, sort components by descending size.

        Returns:
            list[list[str]]: List of connected components with token labels.
        """
        components = self.get_connected_components(
            threshold=threshold, sort_by_size=sort_by_size
        )
        return [[self.labels[idx] for idx in comp] for comp in components]

    def get_block_diagonal_reordering(
        self, threshold: float = 0.05, sort_by_size: bool = True
    ) -> tuple[list[int], list[str]]:
        """
        Order indices such that elements belonging to the same connected component are contiguous.

        Args:
            threshold: Probability threshold tau.
            sort_by_size: If True, sort components by descending size.

        Returns:
            tuple[list[int], list[str]]: (permuted_indices, permuted_labels).
        """
        components = self.get_connected_components(
            threshold=threshold, sort_by_size=sort_by_size
        )
        permuted_indices = [idx for comp in components for idx in comp]
        permuted_labels = [self.labels[idx] for idx in permuted_indices]
        return permuted_indices, permuted_labels

    def get_component_slices(
        self, threshold: float = 0.05, sort_by_size: bool = True, min_size: int = 1
    ) -> list[tuple[int, int]]:
        """
        Compute slice ranges (start, end) for components in the block-diagonal reordered matrix.

        Args:
            threshold: Probability threshold tau.
            sort_by_size: If True, sort components by descending size.
            min_size: Minimum component size to include in slice list (e.g. 2 for non-trivial clusters).

        Returns:
            list[tuple[int, int]]: List of (start_idx, end_idx) half-open intervals.
        """
        components = self.get_connected_components(
            threshold=threshold, sort_by_size=sort_by_size
        )
        slices: list[tuple[int, int]] = []
        offset = 0
        for comp in components:
            size = len(comp)
            if size >= min_size:
                slices.append((offset, offset + size))
            offset += size
        return slices

    def reorder_matrix(
        self, matrix: np.ndarray, permuted_indices: list[int]
    ) -> np.ndarray:
        """
        Reorder rows and columns of a matrix according to a permutation vector.

        Args:
            matrix: 2D numpy array of shape (N, N).
            permuted_indices: List of length N with reordered index permutation.

        Returns:
            np.ndarray: Reordered 2D array.
        """
        arr = np.asarray(matrix)
        if (
            arr.ndim != 2
            or arr.shape[0] != len(permuted_indices)
            or arr.shape[1] != len(permuted_indices)
        ):
            raise ValueError(
                f"Matrix shape {arr.shape} does not match permutation length {len(permuted_indices)}"
            )
        return arr[np.ix_(permuted_indices, permuted_indices)]

    def compute_manifold_stability(
        self,
        snr_levels: list[float],
        cond_probs_by_snr: list[np.ndarray],
        threshold: float = 0.05,
    ) -> dict[str, Any]:
        """
        Compute manifold stability metrics across SNR levels.

        For each SNR level, computes the number of connected components and the maximum
        component diameter (longest shortest-path distance within any component).

        Args:
            snr_levels: List of SNR values in dB.
            cond_probs_by_snr: List of conditional probability matrices corresponding to each SNR.
            threshold: Probability threshold tau.

        Returns:
            dict: Stability metrics containing:
                - 'snr_levels': list of SNR values
                - 'num_components': list of component counts
                - 'component_counts': alias for num_components
                - 'max_diameters': list of max component diameters
                - 'max_component_diameters': alias for max_diameters
                - 'components_by_snr': list of components for each SNR
                - 'threshold': threshold value used
        """
        return self._compute_stability(
            snr_levels=snr_levels,
            cond_probs_by_snr=cond_probs_by_snr,
            threshold=threshold,
            labels=self.labels,
        )

    @classmethod
    def compute_stability(
        cls,
        snr_levels: list[float],
        cond_probs_by_snr: list[np.ndarray],
        threshold: float = 0.05,
        labels: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Classmethod to compute manifold stability across SNR tiers without pre-instantiation.
        """
        return cls._compute_stability(
            snr_levels=snr_levels,
            cond_probs_by_snr=cond_probs_by_snr,
            threshold=threshold,
            labels=labels,
        )

    @staticmethod
    def _compute_stability(
        snr_levels: list[float],
        cond_probs_by_snr: list[np.ndarray],
        threshold: float = 0.05,
        labels: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if len(snr_levels) != len(cond_probs_by_snr):
            raise ValueError(
                f"snr_levels count ({len(snr_levels)}) does not match cond_probs_by_snr count ({len(cond_probs_by_snr)})"
            )

        component_counts: list[int] = []
        max_diameters: list[int] = []
        components_by_snr: list[list[list[int]]] = []

        for snr, mat in zip(snr_levels, cond_probs_by_snr):
            mat_arr = np.asarray(mat, dtype=np.float64)
            n = mat_arr.shape[0]
            lbls = (
                labels
                if labels is not None and len(labels) == n
                else [str(i) for i in range(n)]
            )
            analyzer = PhoneticManifoldAnalyzer(cond_probs=mat_arr, labels=lbls)

            A = analyzer.get_adjacency_matrix(diagonal=0.0)
            comps = analyzer.get_connected_components(
                threshold=threshold, sort_by_size=False
            )
            component_counts.append(len(comps))
            components_by_snr.append(comps)

            # Compute diameter of each component
            comp_diameters = []
            for comp in comps:
                if len(comp) <= 1:
                    comp_diameters.append(0)
                    continue

                # Run BFS from all nodes in comp to find max shortest path
                max_d = 0
                for root in comp:
                    dist = {root: 0}
                    queue = [root]
                    while queue:
                        u = queue.pop(0)
                        for v in comp:
                            if v not in dist and v != u and A[u, v] > threshold:
                                dist[v] = dist[u] + 1
                                queue.append(v)
                    if dist:
                        max_d = max(max_d, max(dist.values()))
                comp_diameters.append(max_d)

            max_diameters.append(max(comp_diameters, default=0))

        return {
            "snr_levels": [float(s) for s in snr_levels],
            "num_components": component_counts,
            "component_counts": component_counts,
            "max_diameters": max_diameters,
            "max_component_diameters": max_diameters,
            "components_by_snr": components_by_snr,
            "threshold": threshold,
        }
