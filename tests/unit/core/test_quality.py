"""质量门范围：只判模型通道，非模型通道的伪迹不得拦下整块。"""

from __future__ import annotations

import numpy as np

from eeg_core.contracts import DEFAULT_CONFIG, MODEL_CHANNELS, EEGChunk
from eeg_core.processing import QualityChecker

AUX_CHANNELS = ("AUX01", "AUX02")
CHANNELS = (*MODEL_CHANNELS, *AUX_CHANNELS)
SAMPLE_RATE = 160.0


def _checker() -> QualityChecker:
    processing = DEFAULT_CONFIG.processing
    return QualityChecker(
        amplitude_limit_v=processing.amplitude_limit_uv * 1e-6,
        flatline_sec=processing.flatline_sec,
        flatline_tolerance_v=processing.flatline_tolerance_uv * 1e-6,
        config=DEFAULT_CONFIG,
    )


def _chunk(
    samples: np.ndarray,
    *,
    seq: int = 0,
    start: int = 0,
    channels: tuple[str, ...] = CHANNELS,
) -> EEGChunk:
    return EEGChunk(
        source_id="synthetic",
        recording_id="S999R04",
        seq=seq,
        start_sample=start,
        sample_rate_hz=SAMPLE_RATE,
        channel_names=channels,
        samples=samples,
        released_monotonic_ms=0.0,
        discontinuity=(seq == 0),
    )


def _noise(n_samples: int, n_channels: int = len(CHANNELS), seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 5e-6, (n_channels, n_samples))


def test_aux_channel_artifacts_do_not_invalidate_block() -> None:
    """额区/辅助通道的眨眼尖峰、NaN、死通道都不该让中央区数据进不了算法。"""
    spike = _noise(320)
    spike[CHANNELS.index("AUX01"), 100] = 800e-6

    nan = _noise(320)
    nan[CHANNELS.index("AUX02"), 100] = np.nan

    flat = _noise(320)
    flat[CHANNELS.index("AUX01"), :] = 0.0  # 2 s 死通道

    for name, samples in (("spike", spike), ("nan", nan), ("flatline", flat)):
        report = _checker().check(_chunk(samples))
        assert report.valid, f"{name} 出现在非模型通道却拦下了整块：{report.reasons}"
        assert report.bad_channels == ()


def test_model_channel_amplitude_marks_only_that_channel() -> None:
    samples = _noise(8)
    samples[CHANNELS.index("Cz"), 3] += 800e-6
    report = _checker().check(_chunk(samples))
    assert report.valid is False
    assert report.reasons == ("amplitude",)
    assert report.bad_channels == ("Cz",)


def test_bad_channels_follow_input_order() -> None:
    samples = _noise(8)
    samples[CHANNELS.index("CP3"), 1] += 800e-6
    samples[CHANNELS.index("FC3"), 5] -= 800e-6
    report = _checker().check(_chunk(samples))
    assert report.bad_channels == ("FC3", "CP3")


def test_model_channel_non_finite() -> None:
    samples = _noise(8)
    samples[CHANNELS.index("FCz"), 2] = np.inf
    report = _checker().check(_chunk(samples))
    assert report.valid is False
    assert report.reasons == ("non_finite",)
    assert report.bad_channels == ("FCz",)


def test_model_channel_flatline_is_detected_after_one_second() -> None:
    checker = _checker()
    reports = []
    for seq, start in enumerate(range(0, 240, 8)):
        samples = _noise(8, seed=seq)
        samples[CHANNELS.index("C4")] = 0.0
        reports.append(checker.check(_chunk(samples, seq=seq, start=start)))

    invalid = [(index, report) for index, report in enumerate(reports) if not report.valid]
    assert invalid, "1 s 常量段没有被判 flatline"
    # 160 Hz × 1 s = 160 个样本；8 样本一块，第 20 块末触发。
    assert invalid[0][0] == 19
    assert invalid[0][1].reasons == ("flatline",)
    assert invalid[0][1].bad_channels == ("C4",)


def test_missing_model_channel_is_reported() -> None:
    channels = tuple(name for name in CHANNELS if name != "Cz")
    report = _checker().check(_chunk(_noise(8, n_channels=len(channels)), channels=channels))
    assert report.valid is False
    assert report.reasons == ("missing_channel",)
    assert report.bad_channels == ("Cz",)
