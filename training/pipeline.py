"""把一条 ``Recording`` 流式送进 A 的 ``FilterBank``，切出带标签的 18 维窗。

这是训练侧对 A→B 边界的唯一入口。三条纪律在这里落地：

* **不在 B 里重做滤波/质量**：只调用 ``FilterBank``，只读 ``quality.valid``；
* **不整段丢进 FilterBank**：按 50 ms（8 样本）小块流式喂，与运行时同一节奏，
  避免一个额区伪迹把整段记录判成 ``invalid``；
* **不跨 reset 拼窗**：A 在 invalid / discontinuity 后清空滤波状态并重新预热，
  窗必须落在同一段连续有效数据里，且窗起点已过 3 s 预热。

标签只在这里出现，来自 ``Recording.markers``，不会进入任何 ``EEGChunk``。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from eeg_core.contracts import (
    AppConfig,
    ContractError,
    EEGChunk,
    ReferenceMarker,
    SourceInfo,
    WindowRange,
)
from eeg_core.processing import (
    FilterBank,
    is_window_usable,
    iter_window_ranges,
    log_variance_features,
    required_warmup_samples,
)

__all__ = [
    "LABEL_BY_MARKER",
    "DatasetStats",
    "RecordingLike",
    "WindowSample",
    "build_feature_dataset",
    "label_of_window",
    "load_split",
]

LABEL_BY_MARKER = {"T0": "idle", "T1": "left_hand", "T2": "right_hand"}
"""EEGMMIDB run 4/8/12 的标签映射；其它 run 的同名标签不能沿用到本项目。"""


class RecordingLike(Protocol):
    """训练侧需要的最小记录接口；``eeg_app.sources.input.Recording`` 满足它。"""

    @property
    def info(self) -> SourceInfo: ...

    @property
    def samples(self) -> np.ndarray: ...

    @property
    def markers(self) -> tuple[ReferenceMarker, ...]: ...

    @property
    def n_samples(self) -> int: ...


@dataclass(frozen=True, slots=True)
class WindowSample:
    """一个可用窗：定位、18 维特征、参考标签。"""

    subject_id: str | None
    recording_id: str
    label: str
    window: WindowRange
    features: np.ndarray

    def __post_init__(self) -> None:
        features = np.array(self.features, dtype=np.float64, copy=True)
        features.flags.writeable = False
        object.__setattr__(self, "features", features)


@dataclass(slots=True)
class DatasetStats:
    """数据产出报告：A 的质量门和预热血统实际留下了多少窗。"""

    recording_id: str
    n_samples: int
    n_chunks: int = 0
    invalid_chunks: int = 0
    total_windows: int = 0
    usable_windows: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "recording_id": self.recording_id,
            "n_samples": self.n_samples,
            "n_chunks": self.n_chunks,
            "invalid_chunks": self.invalid_chunks,
            "total_windows": self.total_windows,
            "usable_windows": self.usable_windows,
            "class_counts": dict(self.class_counts),
        }


def _make_filter_bank(config: AppConfig) -> FilterBank:
    """按契约把 µV 阈值换算成伏，换算只做一次。"""
    return FilterBank(
        amplitude_limit_v=config.processing.amplitude_limit_uv * 1e-6,
        flatline_sec=config.processing.flatline_sec,
        flatline_tolerance_v=config.processing.flatline_tolerance_uv * 1e-6,
        config=config,
    )


def _stream_bands(
    recording: RecordingLike,
    config: AppConfig,
    *,
    filter_bank: FilterBank | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DatasetStats]:
    """把整条记录流式过 ``FilterBank``，缓存两路频带与有效/段掩码。"""
    info = recording.info
    samples = np.asarray(recording.samples)
    if samples.ndim != 2:
        raise ContractError(f"samples 必须是二维数组，收到 {samples.ndim} 维")
    if samples.shape[1] != recording.n_samples:
        raise ContractError(
            f"samples 样本数与 n_samples 不一致：{samples.shape[1]} vs {recording.n_samples}"
        )
    if len(info.channel_names) != samples.shape[0]:
        raise ContractError("通道表长度与 samples 行数不一致")
    missing = [name for name in config.source.model_channels if name not in info.channel_names]
    if missing:
        raise ContractError("记录缺少模型通道：" + "、".join(missing))

    n_samples = int(recording.n_samples)
    n_channels = int(samples.shape[0])
    band_first = np.full((n_channels, n_samples), np.nan, dtype=np.float64)
    band_second = np.full((n_channels, n_samples), np.nan, dtype=np.float64)
    valid = np.zeros(n_samples, dtype=bool)
    segment_start = np.full(n_samples, -1, dtype=np.int64)

    chunk_samples = max(
        1,
        int(round(config.source.target_chunk_ms / 1000.0 * info.sample_rate_hz)),
    )
    bank = filter_bank if filter_bank is not None else _make_filter_bank(config)
    stats = DatasetStats(recording_id=info.recording_id, n_samples=n_samples)
    segment: int | None = None

    for seq, start in enumerate(range(0, n_samples, chunk_samples)):
        stop = min(start + chunk_samples, n_samples)
        chunk = EEGChunk(
            source_id=info.source_id,
            recording_id=info.recording_id,
            seq=seq,
            start_sample=start,
            sample_rate_hz=info.sample_rate_hz,
            channel_names=info.channel_names,
            samples=samples[:, start:stop],
            released_monotonic_ms=0.0,
            discontinuity=(seq == 0),
        )
        processed = bank.process(chunk)
        stats.n_chunks += 1
        quality = processed["quality"]
        if not quality.valid:
            stats.invalid_chunks += 1
            segment = None
            continue
        if chunk.discontinuity or segment is None:
            segment = start

        first = processed["band_8_13_hz"]
        second = processed["band_13_30_hz"]
        if first is None or second is None:
            raise ContractError("FilterBank 报告有效但未返回频带数组")
        band_first[:, start:stop] = first
        band_second[:, start:stop] = second
        valid[start:stop] = True
        segment_start[start:stop] = segment

    return band_first, band_second, valid, segment_start, stats


def label_of_window(
    markers: Sequence[ReferenceMarker],
    start_sample: int,
    end_sample: int,
) -> str | None:
    """窗口完整落在某个标注区间内才给标签；跨转换区的一律丢弃。

    严格包含比“按窗中心贴标签”少一些样本，但不会把转换区混进训练集。评估阶段
    另按 PRD §10 排除标签转换前后各 1 s。
    """
    for marker in markers:
        if marker.duration_samples is None:
            continue
        stop = marker.onset_sample + marker.duration_samples
        if marker.onset_sample <= start_sample and end_sample <= stop:
            return LABEL_BY_MARKER.get(marker.label)
    return None


def build_feature_dataset(
    recording: RecordingLike,
    config: AppConfig,
    *,
    filter_bank: FilterBank | None = None,
) -> tuple[list[WindowSample], DatasetStats]:
    """一条记录 → 带标签的 18 维窗列表 + 数据产出报告。"""
    band_first, band_second, valid, segment_start, stats = _stream_bands(
        recording, config, filter_bank=filter_bank
    )
    indices = [recording.info.channel_names.index(name) for name in config.source.model_channels]
    warmup = required_warmup_samples(config)
    subject_id = recording.info.subject_id
    samples: list[WindowSample] = []

    for window in iter_window_ranges(recording.n_samples, config):
        stats.total_windows += 1
        if not is_window_usable(window, valid, segment_start, warmup=warmup):
            continue
        label = label_of_window(
            recording.markers, window.start_sample, window.end_sample
        )
        if label is None:
            continue
        features = log_variance_features(
            band_first,
            band_second,
            indices,
            window.start_sample,
            window.end_sample,
            floor=config.processing.log_variance_floor,
        )
        if features.shape != (len(indices) * 2,):
            raise ContractError(
                f"特征维数应为 {len(indices) * 2}，收到 {features.shape}"
            )
        samples.append(
            WindowSample(
                subject_id=subject_id,
                recording_id=recording.info.recording_id,
                label=label,
                window=window,
                features=features,
            )
        )

    stats.usable_windows = len(samples)
    stats.class_counts = dict(Counter(item.label for item in samples))
    return samples, stats


def load_split(path: str | Path) -> Mapping[str, object]:
    """读取 ``training/splits.json``；只做 JSON 解析，不替调用方猜字段。"""
    import json

    split_path = Path(path)
    if not split_path.is_file():
        raise ContractError(f"找不到划分文件：{split_path}")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError("划分文件顶层必须是 JSON 对象")
    return data
