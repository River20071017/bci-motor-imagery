"""T05：同一批样本按不同块长供给，A 的两路频带应一致。

这是训练/回放能对上的前提：如果 ``FilterBank`` 的结果随块长变化，训练用的 8 样本块
和回放/运行时用别的块长就会得到不同特征。
"""

from __future__ import annotations

import numpy as np

from eeg_core.contracts import DEFAULT_CONFIG, MODEL_CHANNELS, EEGChunk
from eeg_core.processing import FilterBank


def _samples(n_samples: int = 1280) -> np.ndarray:
    time = np.arange(n_samples) / 160.0
    return np.stack(
        [
            (index + 1) * 5e-6 * np.sin(2 * np.pi * (8 + index) * time)
            for index in range(len(MODEL_CHANNELS))
        ]
    )


def _chunk(samples: np.ndarray, start: int, stop: int, seq: int) -> EEGChunk:
    return EEGChunk(
        source_id="synthetic",
        recording_id="S999R04",
        seq=seq,
        start_sample=start,
        sample_rate_hz=160.0,
        channel_names=MODEL_CHANNELS,
        samples=samples[:, start:stop],
        released_monotonic_ms=0.0,
        discontinuity=(seq == 0),
    )


def _run(samples: np.ndarray, chunk_samples: int) -> tuple[np.ndarray, np.ndarray]:
    bank = FilterBank(
        amplitude_limit_v=DEFAULT_CONFIG.processing.amplitude_limit_uv * 1e-6,
        flatline_sec=DEFAULT_CONFIG.processing.flatline_sec,
        flatline_tolerance_v=DEFAULT_CONFIG.processing.flatline_tolerance_uv * 1e-6,
        config=DEFAULT_CONFIG,
    )
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for seq, start in enumerate(range(0, samples.shape[1], chunk_samples)):
        stop = min(start + chunk_samples, samples.shape[1])
        result = bank.process(_chunk(samples, start, stop, seq))
        assert result["quality"].valid
        assert result["band_8_13_hz"] is not None
        assert result["band_13_30_hz"] is not None
        first.append(result["band_8_13_hz"])
        second.append(result["band_13_30_hz"])
    return np.concatenate(first, axis=1), np.concatenate(second, axis=1)


def test_chunk_size_does_not_change_band_arrays() -> None:
    samples = _samples()
    first_8, second_8 = _run(samples, 8)
    first_80, second_80 = _run(samples, 80)
    assert np.allclose(first_8, first_80, atol=1e-15, rtol=0)
    assert np.allclose(second_8, second_80, atol=1e-15, rtol=0)
