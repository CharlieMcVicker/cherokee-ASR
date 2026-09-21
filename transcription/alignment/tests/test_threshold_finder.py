"""
Unit tests for interactive CLI binary search tool for alignment cost thresholding.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
import pytest

from transcription.alignment.threshold_finder import (
    AlignmentRecord,
    AlignmentThresholdFinder,
    ThresholdMetrics,
    _extract_record_from_dict,
    find_threshold_bounds,
    load_alignment_records,
    main,
    parse_verse_reference,
)


@pytest.fixture
def sample_records() -> List[AlignmentRecord]:
    """Create a diverse synthetic sample of alignment records for testing."""
    records = []
    # 20 records with costs linearly spaced from 0.02 to 0.40
    for i in range(20):
        cost = round(0.02 + i * 0.02, 4)
        verse_num = i + 1
        line_id = f"0201{verse_num:02d}"  # Mark 01:XX
        records.append(
            AlignmentRecord(
                verse_id=line_id,
                cherokee_syllabary=f"Cherokee Syllabary {verse_num}",
                text=f"Phonetic Text {verse_num}",
                emitted_text=f"ASR Hypothesis {verse_num}",
                cost=cost,
                audio_path=f"audio/mark_01_{verse_num:02d}.wav",
                start_sec=float(i * 5),
                end_sec=float(i * 5 + 4),
                english=f"English verse {verse_num}",
            )
        )
    return records


class TestVerseIdParsing:
    """Test human-readable verse ID parsing."""

    def test_parse_6digit_standard_bible_ids(self) -> None:
        assert parse_verse_reference("020114") == "Mark 01:14 (020114)"
        assert parse_verse_reference("010503") == "Matthew 05:03 (010503)"
        assert parse_verse_reference("040316") == "John 03:16 (040316)"
        assert parse_verse_reference("272221") == "Revelation 22:21 (272221)"

    def test_parse_underscore_pattern_ids(self) -> None:
        assert parse_verse_reference("mark_01_14") == "Mark 01:14 (mark_01_14)"
        assert parse_verse_reference("matthew_05_03") == "Matthew 05:03 (matthew_05_03)"

    def test_parse_arbitrary_ids(self) -> None:
        assert parse_verse_reference("chunk_042") == "chunk_042"
        assert parse_verse_reference("") == "Unknown"


class TestRecordIngestion:
    """Test loading alignment records from varied data formats and sources."""

    def test_extract_record_from_dict_aliases(self) -> None:
        raw_manifest_item = {
            "line_id": "020101",
            "cherokee_syllabary": "ᎠᏓᎴᏂᏍᎬ",
            "text": "A-da-le-ni-s-gv",
            "emitted_text": "atalenihskv",
            "cer": 0.0938,
            "start": 8.26,
            "end": 14.85,
        }
        rec = _extract_record_from_dict(raw_manifest_item, default_audio="audio.mp3")
        assert rec is not None
        assert rec.verse_id == "020101"
        assert rec.cherokee_syllabary == "ᎠᏓᎴᏂᏍᎬ"
        assert rec.cost == 0.0938
        assert rec.audio_path == "audio.mp3"
        assert rec.duration_sec == round(14.85 - 8.26, 3)

    def test_extract_record_missing_cost_returns_none(self) -> None:
        bad_item = {"line_id": "020101", "text": "hello"}
        rec = _extract_record_from_dict(bad_item)
        assert rec is None

    def test_load_from_manifest_dict(self) -> None:
        manifest_data = {
            "audio_source": "/path/to/mark_01.mp3",
            "lines": [
                {
                    "line_id": "020101",
                    "cherokee_syllabary": "ᎠᏓᎴᏂᏍᎬ",
                    "text": "A-da-le-ni-s-gv",
                    "emitted_text": "atalenihskv",
                    "cer": 0.09,
                    "start": 8.0,
                    "end": 14.0,
                },
                {
                    "line_id": "020102",
                    "cherokee_syllabary": "ᎾᏍᎩᏯ",
                    "text": "Na-s-gi-ya",
                    "emitted_text": "nvhskayahi'a",
                    "cer": 0.12,
                    "start": 16.0,
                    "end": 33.0,
                },
            ],
        }
        records, files = load_alignment_records(manifest_data)
        assert len(records) == 2
        assert records[0].audio_path == "/path/to/mark_01.mp3"
        assert records[1].cost == 0.12

    def test_load_from_json_file(self, tmp_path: Path) -> None:
        json_file = tmp_path / "test_manifest.json"
        manifest_data = {
            "audio_source": "audio.wav",
            "lines": [
                {"line_id": "010101", "cer": 0.05, "text": "v1"},
                {"line_id": "010102", "cer": 0.15, "text": "v2"},
            ],
        }
        json_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        records, files = load_alignment_records(json_file)
        assert len(records) == 2
        assert len(files) == 1
        assert str(json_file) in files[0]

    def test_load_from_csv_file(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test_records.csv"
        csv_content = (
            "verse_id,cherokee_syllabary,text,emitted_text,cost,audio_path,start,end\n"
            "020101,ᎠᏓᎴᏂᏍᎬ,A-da-le-ni-s-gv,atalenihskv,0.08,mark_01.mp3,1.0,4.0\n"
            "020102,ᎾᏍᎩᏯ,Na-s-gi-ya,nvhskayahi,0.14,mark_01.mp3,5.0,9.0\n"
        )
        csv_file.write_text(csv_content, encoding="utf-8")

        records, files = load_alignment_records(csv_file)
        assert len(records) == 2
        assert records[0].verse_id == "020101"
        assert records[0].cost == 0.08
        assert records[1].cost == 0.14

    def test_load_from_directory(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "book1"
        dir2 = tmp_path / "book2"
        dir1.mkdir()
        dir2.mkdir()

        m1 = {"audio_source": "b1.wav", "lines": [{"line_id": "010101", "cer": 0.05}]}
        m2 = {"audio_source": "b2.wav", "lines": [{"line_id": "020101", "cer": 0.10}]}
        (dir1 / "alignment_manifest.json").write_text(json.dumps(m1), encoding="utf-8")
        (dir2 / "alignment_manifest.json").write_text(json.dumps(m2), encoding="utf-8")

        records, files = load_alignment_records(tmp_path)
        assert len(records) == 2
        assert len(files) == 2


class TestBinarySearchAlgorithm:
    """Test binary search bracket narrowing and sampling logic."""

    def test_initialization_bounds(self, sample_records: List[AlignmentRecord]) -> None:
        finder = AlignmentThresholdFinder(sample_records)
        assert finder.t_low == pytest.approx(0.02)
        assert finder.t_high == pytest.approx(0.40)
        assert len(finder.costs) == 20

    def test_sampling_closest_to_midpoint(
        self, sample_records: List[AlignmentRecord]
    ) -> None:
        finder = AlignmentThresholdFinder(sample_records, k_samples=3)
        # Midpoint of [0.02, 0.40] is 0.21
        samples = finder.sample_near_cost(0.21, k=3)
        assert len(samples) == 3
        # Closest costs to 0.21 should be 0.20, 0.22, 0.24 or 0.18
        costs = [s.cost for s in samples]
        assert 0.20 in costs or 0.22 in costs

    def test_simulated_interactive_search_converges_to_target(
        self, sample_records: List[AlignmentRecord]
    ) -> None:
        """
        Simulate a user who accepts any alignment with cost <= 0.15 and rejects above 0.15.
        Verify that binary search converges to ~0.15.
        """
        target_quality_boundary = 0.15
        finder = AlignmentThresholdFinder(sample_records, tolerance=0.01, max_iter=15)

        def mock_user_response(
            state: Dict[str, Any], samples: List[AlignmentRecord]
        ) -> str:
            t_mid = state["t_mid"]
            if t_mid <= target_quality_boundary:
                return "accept"
            else:
                return "reject"

        metrics = finder.run(prompt_callback=mock_user_response, quiet=True)

        assert metrics.threshold == pytest.approx(target_quality_boundary, abs=0.02)
        assert metrics.iterations > 0
        assert metrics.accepted_count == sum(
            1 for r in sample_records if r.cost <= metrics.threshold
        )

    def test_command_unsure_resamples(
        self, sample_records: List[AlignmentRecord]
    ) -> None:
        finder = AlignmentThresholdFinder(sample_records, k_samples=2, tolerance=0.01)
        calls = []

        def callback(state: Dict[str, Any], samples: List[AlignmentRecord]) -> str:
            calls.append([s.verse_id for s in samples])
            if len(calls) == 1:
                return "unsure"
            return "done"

        metrics = finder.run(prompt_callback=callback, quiet=True)
        assert len(calls) == 2
        # Ensure second call skipped previously sampled items if possible
        assert calls[0] != calls[1]
        assert any(step["action"] == "unsure" for step in metrics.history)

    def test_command_set_manual_threshold(
        self, sample_records: List[AlignmentRecord]
    ) -> None:
        finder = AlignmentThresholdFinder(sample_records)

        def callback(state: Dict[str, Any], samples: List[AlignmentRecord]) -> str:
            return "set 0.125"

        metrics = finder.run(prompt_callback=callback, quiet=True)
        assert metrics.threshold == pytest.approx(0.125)
        assert metrics.t_low == pytest.approx(0.125)
        assert metrics.t_high == pytest.approx(0.125)

    def test_command_done_early_exit(
        self, sample_records: List[AlignmentRecord]
    ) -> None:
        finder = AlignmentThresholdFinder(sample_records)

        def callback(state: Dict[str, Any], samples: List[AlignmentRecord]) -> str:
            return "done"

        metrics = finder.run(prompt_callback=callback, quiet=True)
        assert metrics.iterations == 1
        assert metrics.threshold == pytest.approx((0.02 + 0.40) / 2.0)


class TestHeadlessAndHelperModes:
    """Test headless overrides, quantile calculations, and programmatic helper functions."""

    def test_headless_direct_threshold(
        self, sample_records: List[AlignmentRecord]
    ) -> None:
        finder = AlignmentThresholdFinder(sample_records)
        metrics = finder.run(threshold=0.10, quiet=True)
        assert metrics.threshold == pytest.approx(0.10)
        # In sample_records (costs: 0.02, 0.04, 0.06, 0.08, 0.10, ...), 5 items are <= 0.10
        assert metrics.accepted_count == 5
        assert metrics.rejected_count == 15
        assert metrics.acceptance_rate == pytest.approx(0.25)
        assert metrics.percentile == pytest.approx(25.0)

    def test_headless_quantile(self, sample_records: List[AlignmentRecord]) -> None:
        finder = AlignmentThresholdFinder(sample_records)
        metrics = finder.run(quantile=0.80, quiet=True)
        # 80th percentile of 20 items is 16th item (0.02 + 15*0.02 = 0.32)
        assert metrics.threshold == pytest.approx(0.32)
        assert metrics.accepted_count >= 16

    def test_find_threshold_bounds_helper(
        self, sample_records: List[AlignmentRecord]
    ) -> None:
        thresh, metrics = find_threshold_bounds(
            sample_records, target_acceptance_rate=0.50
        )
        assert thresh > 0.0
        assert metrics.acceptance_rate >= 0.50


class TestExportAndMetricsSummary:
    """Test summary statistics and JSON export integrity."""

    def test_export_results_json(
        self, sample_records: List[AlignmentRecord], tmp_path: Path
    ) -> None:
        finder = AlignmentThresholdFinder(sample_records)
        metrics = finder.run(threshold=0.15, quiet=True)

        out_path = tmp_path / "runs" / "evaluation" / "alignment_threshold.json"
        exported_file = finder.export_results(metrics, output_path=out_path)

        assert exported_file.exists()
        with open(exported_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["threshold"] == pytest.approx(0.15)
        assert data["total_verses"] == 20
        assert "accepted_count" in data
        assert "rejected_count" in data
        assert "acceptance_rate" in data
        assert "percentile" in data
        assert "mean_accepted_cost" in data
        assert "mean_rejected_cost" in data
        assert "timestamp" in data
        assert "cost_min" in data
        assert "cost_max" in data


class TestCliEntryPoint:
    """Test CLI main() execution with argument combinations."""

    def test_cli_headless_threshold(self, tmp_path: Path) -> None:
        input_file = tmp_path / "records.json"
        manifest_data = {
            "audio_source": "audio.wav",
            "lines": [
                {"line_id": "020101", "cer": 0.04},
                {"line_id": "020102", "cer": 0.08},
                {"line_id": "020103", "cer": 0.16},
            ],
        }
        input_file.write_text(json.dumps(manifest_data), encoding="utf-8")
        out_file = tmp_path / "out_threshold.json"

        ret = main(
            [
                "--input",
                str(input_file),
                "--output",
                str(out_file),
                "--threshold",
                "0.08",
                "--quiet",
            ]
        )
        assert ret == 0
        assert out_file.exists()
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["threshold"] == pytest.approx(0.08)
        assert data["accepted_count"] == 2
        assert data["rejected_count"] == 1

    def test_cli_headless_quantile(self, tmp_path: Path) -> None:
        input_file = tmp_path / "records.json"
        manifest_data = {
            "audio_source": "audio.wav",
            "lines": [
                {"line_id": "020101", "cer": 0.05},
                {"line_id": "020102", "cer": 0.10},
            ],
        }
        input_file.write_text(json.dumps(manifest_data), encoding="utf-8")
        out_file = tmp_path / "out_threshold.json"

        ret = main(
            [
                "--input",
                str(input_file),
                "--output",
                str(out_file),
                "--quantile",
                "0.50",
                "--quiet",
            ]
        )
        assert ret == 0
        assert out_file.exists()

    def test_cli_missing_input_error(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "missing.json"
        ret = main(["--input", str(non_existent), "--threshold", "0.10"])
        assert ret == 1
