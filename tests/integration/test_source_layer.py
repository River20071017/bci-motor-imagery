"""数据源层：MNE 统一读取、通道别名、显式重采样，以及“通道数不再进配置”的回归。

契约 1.1 的核心变化：记录有多少通道是数据源自己的事，``SourceConfig`` 只描述算法要求
（目标采样率、模型通道、重参考）。这里用内存里的 ``mne.io.RawArray`` 覆盖 22 导/250 Hz
的 BCIC IV 形态和 2 导消费级设备的通道集，不需要真实数据文件。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from eeg_app.sources import (
    Recording,
    ReplaySource,
    apply_channel_aliases,
    recording_from_raw,
)
from eeg_core.contracts import (
    DEFAULT_CONFIG,
    ContractError,
    EEGChunk,
    IncompatibleSourceError,
    SourceInfo,
)

MODEL_CHANNELS = DEFAULT_CONFIG.source.model_channels
BCIC_CHANNELS = (
    "Fz",
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
    "O1",
)
"""BCIC IV 2a 的 22 导形态；是否含 FCz 以实际文件头为准，这里用近似集合验证“不再卡 64”."""


def _raw(channels: tuple[str, ...], *, seconds: float = 2.0, rate: float = 250.0):
    import mne  # noqa: PLC0415 - 只在真正需要构造 Raw 时导入

    info = mne.create_info(list(channels), rate, "eeg")
    data = np.zeros((len(channels), int(round(seconds * rate))))
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_arbitrary_channel_set_is_a_valid_recording() -> None:
    """22 导、250 Hz、含 AF7/AF8 的记录：数据源自描述，不依赖配置通道数。"""
    channels = ("AF7", "AF8", *BCIC_CHANNELS)
    recording = recording_from_raw(_raw(channels), source_id="synthetic", recording_id="A01T")
    assert len(recording.info.channel_names) == 24
    assert recording.info.sample_rate_hz == 250.0
    assert recording.samples.shape == (24, 500)
    assert recording.info.device_metadata["reader"] == "mne"


def test_eeg_chunk_accepts_device_specific_names() -> None:
    """EEGChunk 只校验结构：AF7/AF8 这类设备通道不需要是模型通道。"""
    chunk = EEGChunk(
        source_id="xmuse",
        recording_id="MUSE01",
        seq=0,
        start_sample=0,
        sample_rate_hz=256.0,
        channel_names=("AF7", "AF8"),
        samples=np.zeros((2, 8)),
        released_monotonic_ms=0.0,
        discontinuity=True,
    )
    assert chunk.n_channels == 2
    assert chunk.channel_names == ("AF7", "AF8")


def test_alias_table_applies_before_standardization() -> None:
    names = ("Fc3.", "C3..", "EEG-Cz", "AF7")
    assert apply_channel_aliases(names, {"EEG-Cz": "Cz"}) == ("FC3", "C3", "Cz", "AF7")
    # 没有别名时，PhysioNet 写法仍然能被规范化
    assert apply_channel_aliases(("Fc3.", "Cz.")) == ("FC3", "Cz")


def test_resample_is_explicit_and_recorded() -> None:
    raw = _raw(("C3", "C4", "Cz"), seconds=2.0, rate=250.0)
    recording = recording_from_raw(
        raw, source_id="synthetic", recording_id="A01T", resample_to=160.0
    )
    assert recording.info.sample_rate_hz == 160.0
    assert recording.samples.shape == (3, 320)  # 2 s × 160 Hz
    assert recording.info.device_metadata["original_sample_rate_hz"] == "250"
    assert recording.info.device_metadata["resampled_to_hz"] == "160"


def test_unknown_format_is_rejected_with_clear_error() -> None:
    from pathlib import Path

    from eeg_app.sources import read_raw

    with pytest.raises(ContractError, match="不支持的文件格式"):
        read_raw(Path("_no_such_dir") / "recording.xyz")


def _recording_with(channels: tuple[str, ...]) -> Recording:
    info = SourceInfo(
        source_id="synthetic",
        recording_id="S099R04",
        sample_rate_hz=160.0,
        channel_names=channels,
        n_samples=1600,
    )
    return Recording(info=info, samples=np.zeros((len(channels), 1600)))


def test_replay_no_longer_requires_a_fixed_channel_count() -> None:
    """12 导（含 9 个模型通道）也能回放；通道数不再与 64 绑定。"""
    recording = _recording_with((*MODEL_CHANNELS, "AF7", "AF8", "X1"))
    config = replace(DEFAULT_CONFIG, source=replace(DEFAULT_CONFIG.source, replay_speed=500.0))
    source = ReplaySource(recording, config=config)
    chunk = source.read_chunk()
    assert chunk is not None
    assert chunk.samples.shape == (12, 8)
    source.stop()


def test_replay_still_reports_missing_model_channel() -> None:
    """模型要求的通道缺失仍然是显式错误——但这是 model 要求，不是 chunk 限制。"""
    recording = _recording_with(("C3", "C4", "Cz", "AF7"))
    with pytest.raises(IncompatibleSourceError, match="缺少模型必需通道"):
        ReplaySource(recording, config=DEFAULT_CONFIG)
