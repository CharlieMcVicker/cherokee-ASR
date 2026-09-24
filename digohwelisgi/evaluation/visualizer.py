# -*- coding: utf-8 -*-
"""
digohwelisgi.evaluation.visualizer

Visualization suite for phonetic manifold analysis, confusion heatmaps,
SNR drift diagnostics, and 3D confusion probability meshes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class ManifoldVisualizer:
    """
    Visualization engine for empirical ASR confusion matrices and phonetic manifolds.

    Renders 2D block-diagonal heatmaps, SNR-dependent cluster drift curves, and
    3D surface meshes of conditional confusion probabilities.
    """

    @staticmethod
    def plot_side_by_side(
        confusion_matrix: np.ndarray,
        cost_matrix: np.ndarray,
        labels: list[str],
        save_path: Union[str, Path],
        component_slices: Optional[list[tuple[int, int]]] = None,
        title: Optional[str] = None,
    ) -> Path:
        """
        Generate side-by-side 2D heatmaps of confusion probabilities and substitution costs.

        Highlights block-diagonal connected components with bounding box overlays.

        Args:
            confusion_matrix: (N, N) array of conditional confusion probabilities P(hyp|ref).
            cost_matrix: (N, N) array of normalized substitution costs d(i, j).
            labels: List of N token labels for tick annotations.
            save_path: Path where rendered figure will be saved.
            component_slices: Optional list of (start, end) index tuples marking component blocks.
            title: Optional overall figure suptitle.

        Returns:
            Path: Absolute or resolved path to the saved figure artifact.
        """
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(labels)
        conf_arr = np.asarray(confusion_matrix, dtype=np.float64)
        cost_arr = np.asarray(cost_matrix, dtype=np.float64)

        if conf_arr.shape != (n, n):
            raise ValueError(
                f"confusion_matrix shape {conf_arr.shape} does not match label count ({n}, {n})"
            )
        if cost_arr.shape != (n, n):
            raise ValueError(
                f"cost_matrix shape {cost_arr.shape} does not match label count ({n}, {n})"
            )

        fig_width = max(14.0, n * 0.45)
        fig_height = max(6.5, n * 0.35)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

        # 1. Left: Confusion Probability Heatmap
        im1 = ax1.imshow(
            conf_arr, cmap="Blues", interpolation="nearest", aspect="equal"
        )
        ax1.set_title(
            "Empirical Confusion Probabilities $P(\\mathrm{Hyp} \\mid \\mathrm{Ref})$",
            fontsize=12,
            pad=10,
        )
        ax1.set_xlabel("Hypothesis Token", fontsize=10)
        ax1.set_ylabel("Reference Token", fontsize=10)
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Conditional Probability", fontsize=9)

        # 2. Right: Substitution Cost Heatmap
        im2 = ax2.imshow(
            cost_arr, cmap="YlOrRd", interpolation="nearest", aspect="equal"
        )
        ax2.set_title("Substitution Cost Matrix $d(i, j)$", fontsize=12, pad=10)
        ax2.set_xlabel("Hypothesis Token", fontsize=10)
        ax2.set_ylabel("Reference Token", fontsize=10)
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Normalized Cost", fontsize=9)

        # Set tick labels on both subplots
        ticks = np.arange(n)
        for ax in (ax1, ax2):
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(
                labels,
                rotation=45 if n <= 25 else 90,
                ha="right",
                fontsize=9 if n <= 25 else 7,
            )
            ax.set_yticklabels(labels, fontsize=9 if n <= 25 else 7)

        # Add block-diagonal component bounding boxes if provided
        if component_slices:
            for start, end in component_slices:
                if end > start:
                    width = end - start
                    # Rectangle from (start - 0.5, start - 0.5) spanning width x width
                    rect1 = patches.Rectangle(
                        (start - 0.5, start - 0.5),
                        width,
                        width,
                        fill=False,
                        edgecolor="#e74c3c",
                        linewidth=2.0,
                        linestyle="--",
                    )
                    ax1.add_patch(rect1)

                    rect2 = patches.Rectangle(
                        (start - 0.5, start - 0.5),
                        width,
                        width,
                        fill=False,
                        edgecolor="#2980b9",
                        linewidth=2.0,
                        linestyle="--",
                    )
                    ax2.add_patch(rect2)

        if title:
            fig.suptitle(title, fontsize=14, y=0.98)

        fig.tight_layout()
        fig.savefig(target_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return target_path

    @staticmethod
    def plot_snr_drift(
        snr_levels: list[float],
        component_counts: list[int],
        cer_scores: Optional[list[float]] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
    ) -> Path:
        """
        Generate diagnostic curve of phonetic cluster coalescence / manifold stability across SNR tiers.

        Args:
            snr_levels: List of SNR values in dB (e.g. [30.0, 20.0, 10.0, 5.0, 0.0]).
            component_counts: Number of connected components at each SNR level.
            cer_scores: Optional CER error rates corresponding to each SNR level.
            save_path: Path to save the output plot (default: 'snr_drift.png').
            title: Optional plot title.

        Returns:
            Path: Path to the saved figure artifact.
        """
        target_path = Path(save_path or "snr_drift.png")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if len(snr_levels) != len(component_counts):
            raise ValueError(
                f"snr_levels count ({len(snr_levels)}) does not match component_counts count ({len(component_counts)})"
            )
        if cer_scores is not None and len(cer_scores) != len(snr_levels):
            raise ValueError(
                f"cer_scores count ({len(cer_scores)}) does not match snr_levels count ({len(snr_levels)})"
            )

        fig, ax1 = plt.subplots(figsize=(8.5, 5.0))

        color1 = "#2980b9"
        ax1.set_xlabel("Signal-to-Noise Ratio (SNR dB)", fontsize=11)
        ax1.set_ylabel("Connected Components ($k$)", color=color1, fontsize=11)
        line1 = ax1.plot(
            snr_levels,
            component_counts,
            color=color1,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label="Cluster Count ($k$)",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle=":", alpha=0.6)

        if cer_scores is not None:
            ax2 = ax1.twinx()
            color2 = "#c0392b"
            ax2.set_ylabel("Character Error Rate (CER)", color=color2, fontsize=11)
            line2 = ax2.plot(
                snr_levels,
                cer_scores,
                color=color2,
                marker="s",
                linestyle="--",
                linewidth=2.0,
                markersize=6,
                label="CER",
            )
            ax2.tick_params(axis="y", labelcolor=color2)

            lines = line1 + line2
            labels_leg = [str(line.get_label()) for line in lines]
            ax1.legend(lines, labels_leg, loc="best", framealpha=0.9)
        else:
            ax1.legend(loc="best", framealpha=0.9)

        ax1.set_title(
            title or "Phonetic Manifold Stability & Error Drift across SNR Tiers",
            fontsize=12,
            pad=12,
        )

        fig.tight_layout()
        fig.savefig(target_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return target_path

    @staticmethod
    def plot_3d_confusion_mesh(
        confusion_matrix: np.ndarray,
        labels: list[str],
        save_path: Union[str, Path],
        title: Optional[str] = None,
    ) -> Path:
        """
        Render 3D surface mesh of conditional confusion probabilities.

        Args:
            confusion_matrix: (N, N) array of confusion probabilities.
            labels: List of N token labels.
            save_path: Path where rendered 3D surface plot will be saved.
            title: Optional figure title.

        Returns:
            Path: Path to saved figure artifact.
        """
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(labels)
        conf_arr = np.asarray(confusion_matrix, dtype=np.float64)
        if conf_arr.shape != (n, n):
            raise ValueError(
                f"confusion_matrix shape {conf_arr.shape} does not match label count ({n}, {n})"
            )

        fig = plt.figure(figsize=(10.5, 8.0))
        ax = fig.add_subplot(111, projection="3d")

        x = np.arange(n)
        y = np.arange(n)
        X, Y = np.meshgrid(x, y)
        Z = conf_arr

        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap="viridis",
            edgecolor="k",
            linewidth=0.15,
            alpha=0.88,
            antialiased=True,
        )

        ax.set_xlabel("Hypothesis Index", fontsize=10, labelpad=8)
        ax.set_ylabel("Reference Index", fontsize=10, labelpad=8)
        ax.set_zlabel(
            "Probability $P(\\mathrm{Hyp} \\mid \\mathrm{Ref})$",
            fontsize=10,
            labelpad=8,
        )

        if n <= 20:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
        else:
            step = max(1, n // 10)
            tick_indices = list(range(0, n, step))
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([labels[i] for i in tick_indices], fontsize=8)
            ax.set_yticks(tick_indices)
            ax.set_yticklabels([labels[i] for i in tick_indices], fontsize=8)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.1)
        cbar.set_label("Conditional Probability", fontsize=9)

        ax.set_title(
            title or "3D Phonetic Confusion Probability Surface",
            fontsize=13,
            pad=15,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Tight layout not applied.*")
            try:
                fig.tight_layout()
            except Exception:
                pass
        fig.savefig(target_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return target_path

    @staticmethod
    def plot_regime_comparison(
        regimes: list[dict[str, Any]],
        labels: list[str],
        save_path: Union[str, Path],
        title: Optional[
            str
        ] = "Phonetic Confusion & Cost Manifolds Across Noise Regimes",
    ) -> Path:
        """
        Generate a multi-panel comparison figure across multiple noise regimes (e.g. Low, Mid, High).

        Renders an (R x 2) subplot grid where each row represents a regime:
        - Column 1: Confusion Probability Heatmap P(Hyp | Ref)
        - Column 2: Normalized Substitution Cost Matrix d(i, j)

        Args:
            regimes: List of dicts, each with:
                     - 'name': Regime title (e.g. 'Low Noise (25, 15 dB)')
                     - 'confusion_matrix': (N, N) probability array
                     - 'cost_matrix': (N, N) substitution cost array
                     - 'component_slices': Optional list of (start, end) tuples
            labels: List of N token labels for tick annotations.
            save_path: Path where rendered comparison figure will be saved.
            title: Optional overall figure suptitle.

        Returns:
            Path: Path to saved figure artifact.
        """
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(labels)
        r = len(regimes)
        if r == 0:
            raise ValueError("regimes list must not be empty")

        fig_width = max(14.0, n * 0.45)
        fig_height = max(5.0 * r, n * 0.25 * r)
        fig, axes = plt.subplots(r, 2, figsize=(fig_width, fig_height))
        if r == 1:
            axes = np.expand_dims(axes, 0)

        for row_idx, regime_data in enumerate(regimes):
            regime_name = regime_data.get("name", f"Regime {row_idx + 1}")
            conf_arr = np.asarray(regime_data["confusion_matrix"], dtype=np.float64)
            cost_arr = np.asarray(regime_data["cost_matrix"], dtype=np.float64)
            comp_slices = regime_data.get("component_slices")

            ax1 = axes[row_idx, 0]
            ax2 = axes[row_idx, 1]

            # Confusion heatmap
            im1 = ax1.imshow(
                conf_arr, cmap="Blues", interpolation="nearest", aspect="equal"
            )
            ax1.set_title(
                f"{regime_name} - Confusion $P(\\mathrm{{Hyp}} \\mid \\mathrm{{Ref}})$",
                fontsize=11,
                pad=8,
            )
            ax1.set_xlabel("Hypothesis Token", fontsize=9)
            ax1.set_ylabel("Reference Token", fontsize=9)
            cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label("Probability", fontsize=8)

            # Cost heatmap
            im2 = ax2.imshow(
                cost_arr, cmap="YlOrRd", interpolation="nearest", aspect="equal"
            )
            ax2.set_title(
                f"{regime_name} - Substitution Cost $d(i, j)$", fontsize=11, pad=8
            )
            ax2.set_xlabel("Hypothesis Token", fontsize=9)
            ax2.set_ylabel("Reference Token", fontsize=9)
            cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label("Cost", fontsize=8)

            # Tick annotations
            ticks = np.arange(n)
            for ax in (ax1, ax2):
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels(
                    labels,
                    rotation=45 if n <= 25 else 90,
                    ha="right",
                    fontsize=8 if n <= 25 else 6,
                )
                ax.set_yticklabels(labels, fontsize=8 if n <= 25 else 6)

            # Component overlays
            if comp_slices:
                for start, end in comp_slices:
                    if end > start:
                        width = end - start
                        rect1 = patches.Rectangle(
                            (start - 0.5, start - 0.5),
                            width,
                            width,
                            fill=False,
                            edgecolor="#e74c3c",
                            linewidth=1.5,
                            linestyle="--",
                        )
                        ax1.add_patch(rect1)
                        rect2 = patches.Rectangle(
                            (start - 0.5, start - 0.5),
                            width,
                            width,
                            fill=False,
                            edgecolor="#2980b9",
                            linewidth=1.5,
                            linestyle="--",
                        )
                        ax2.add_patch(rect2)

        if title:
            fig.suptitle(title, fontsize=14, y=0.995)

        fig.tight_layout()
        fig.savefig(target_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return target_path
