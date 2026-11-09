"""训练数据流：合成记录 → FilterBank → 有效段 → 18 维带标签窗。"""

from __future__ import annotations

import numpy as np

from eeg_app.sources.input import Recording
from eeg_core.contracts import DEFAULT_CONFIG, MODEL_CHANNELS, ReferenceMarker, SourceInfo
from training.pipeline import build_feature_dataset, label_of_window


def _recording() -> Recording:
    sample_rate = 160.0
    n_samples = 2240  # 14 s
    time = np.arange(n_samples) / sample_rate
    samples = np.stack(
        [
            (index + 1) * 5e-6 * np.sin(2 * np.pi * (8 + index) * time)
            for index in range(len(MODEL_CHANNELS))
        ]
    )
    info = SourceInfo(
        source_id="synthetic",
        recording_id="S999R04",
        sample_rate_hz=sample_rate,
        channel_names=MODEL_CHANNELS,
        subject_id="S999",
        run="4",
        n_samples=n_samples,
    )
    markers = (
        ReferenceMarker(recording_id="S999R04", onset_sample=0, label="T0", duration_samples=900),
        ReferenceMarker(
            recording_id="S999R04", onset_sample=900, label="T1", duration_samples=500
        ),
        ReferenceMarker(
            recording_id="S999R04", onset_sample=1400, label="T2", duration_samples=840
        ),
    )
    return Recording(info=info, samples=samples, markers=markers)


def test_label_of_window_requires_full_containment() -> None:
    recording = _recording()
    assert label_of_window(recording.markers, 480, 800) == "idle"
    assert label_of_window(recording.markers, 800, 1120) is None  # 跨 T0→T1
    assert label_of_window(recording.markers, 1000, 1320) == "left_hand"


def test_build_feature_dataset_yields_18d_labelled_windows() -> None:
    recording = _recording()
    samples, stats = build_feature_dataset(recording, config=DEFAULT_CONFIG)

    assert stats.n_chunks == 280
    assert stats.invalid_chunks == 0
    assert stats.usable_windows == len(samples)
    assert stats.usable_windows > 0
    assert set(stats.class_counts) == {"idle", "left_hand", "right_hand"}
    assert stats.total_windows >= stats.usable_windows

    for item in samples:
        assert item.features.shape == (18,)
        assert np.isfinite(item.features).all()
        assert item.window.is_aligned(40)
        assert item.recording_id == "S999R04"
        assert item.subject_id == "S999"
