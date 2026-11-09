"""mock EEGChunk 流：生成、落盘、回读、拼回 RecordingLike。"""

from __future__ import annotations

from pathlib import Path

from eeg_core.contracts import DEFAULT_CONFIG, ChunkStreamGuard
from eeg_core.processing import FilterBank
from training.mock import create_mock_chunk_file, make_mock_chunks
from training.pipeline import build_feature_dataset

PATH = Path("training/models/.unit-mock.npz")


def _bank() -> FilterBank:
    return FilterBank(
        amplitude_limit_v=DEFAULT_CONFIG.processing.amplitude_limit_uv * 1e-6,
        flatline_sec=DEFAULT_CONFIG.processing.flatline_sec,
        flatline_tolerance_v=DEFAULT_CONFIG.processing.flatline_tolerance_uv * 1e-6,
        config=DEFAULT_CONFIG,
    )


def test_roundtrip_and_recording_adapter() -> None:
    try:
        stream = create_mock_chunk_file(
            PATH, duration_sec=16.0, trial_sec=4.0, seed=7
        )
        assert stream.info.n_samples == 2560
        assert stream.n_chunks == 320
        assert stream.chunk_samples == 8
        assert [marker.label for marker in stream.markers] == [
            "T0",
            "T1",
            "T2",
            "T0",
        ]

        chunks = list(stream)
        assert len(chunks) == 320
        assert chunks[0].seq == 0
        assert chunks[0].discontinuity is True
        assert chunks[1].start_sample == 8
        assert chunks[0].samples.shape == (64, 8)
        guard = ChunkStreamGuard(stream.info)
        for chunk in chunks:
            guard.validate(chunk)

        recording = stream.as_recording()
        assert recording.samples.shape == (64, 2560)
        assert recording.samples.flags.writeable is False
        assert recording.n_samples == 2560

        samples, stats = build_feature_dataset(recording, DEFAULT_CONFIG)
        assert samples
        assert stats.invalid_chunks == 0
        assert set(stats.class_counts) == {"idle", "left_hand", "right_hand"}
        assert all(item.features.shape == (18,) for item in samples)
    finally:
        PATH.unlink(missing_ok=True)


def test_artifact_makes_quality_invalid() -> None:
    _, _, chunks = make_mock_chunks(duration_sec=8.0, artifact="spike", seed=3)
    bank = _bank()
    results = [bank.process(chunk) for chunk in chunks]
    assert any(not result["quality"].valid for result in results)
    assert any(
        "amplitude" in result["quality"].reasons
        for result in results
        if not result["quality"].valid
    )
