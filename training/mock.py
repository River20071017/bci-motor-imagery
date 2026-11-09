"""按契约生成 mock ``EEGChunk`` 流。

输出是**块级**数据：每条记录被切成固定样本数的 ``EEGChunk``，带
``seq`` / ``start_sample`` / ``released_monotonic_ms`` / ``discontinuity``，
不是 EDF，也不是 ``Recording``。

两种用法：

* 运行时/滤波测试：``load_mock_stream(path)`` 逐块产出 ``EEGChunk``；
* 训练需要整条记录时：``stream.as_recording()`` 把块拼成 ``MockRecording``，
  它实现 ``RecordingLike``，可以直接喂 ``build_feature_dataset``。

信号是确定性的“类正弦 + 噪声”，T1/T2 在 8–13 / 13–30 Hz 上做左右不对称；
只用于跑通链路，不代表真实 EEG 分布。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eeg_core.contracts import (
    CONTRACT_VERSION,
    MODEL_CHANNELS,
    ChunkStreamGuard,
    EEGChunk,
    ReferenceMarker,
    SourceInfo,
)

__all__ = [
    "MOCK_SOURCE_ID",
    "MockChunkStream",
    "MockRecording",
    "create_mock_chunk_file",
    "load_mock_chunks",
    "load_mock_markers",
    "load_mock_stream",
    "make_mock_chunks",
    "save_mock_chunks",
]

MOCK_SOURCE_ID = "mock-eeg"
"""mock 数据的 ``source_id``；不要和 ``physionet-eegmmidb`` 混用。"""

_LEFT = ("FC3", "C3", "CP3")
_RIGHT = ("FC4", "C4", "CP4")
_TRIAL_LABELS = ("T0", "T1", "T2")
_ARTIFACTS = ("none", "spike", "flatline", "nan")


@dataclass(frozen=True, slots=True)
class MockRecording:
    """内存里的整条 mock 记录，满足 ``RecordingLike``。"""

    info: SourceInfo
    samples: np.ndarray
    markers: tuple[ReferenceMarker, ...]

    def __post_init__(self) -> None:
        array = np.asarray(self.samples)
        if array.ndim != 2 or array.shape[0] != len(self.info.channel_names):
            raise ValueError("mock samples 形状必须为 (通道数, 样本数)")
        if array.shape[1] != self.info.n_samples:
            raise ValueError("mock samples 样本数与 info.n_samples 不一致")
        stored = np.array(array, dtype=np.float64, copy=True)
        stored.flags.writeable = False
        object.__setattr__(self, "samples", stored)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[1])


@dataclass(frozen=True, slots=True)
class MockChunkStream:
    """落盘的 chunk 流；``iter(stream)`` 逐块产出 ``EEGChunk``。"""

    path: Path
    info: SourceInfo
    markers: tuple[ReferenceMarker, ...]
    n_chunks: int
    chunk_samples: int

    def __iter__(self) -> Iterator[EEGChunk]:
        return load_mock_chunks(self.path)

    def as_recording(self) -> MockRecording:
        """把块拼回整条记录，供 ``build_feature_dataset`` 使用。"""
        chunks = list(self)
        if not chunks:
            raise ValueError("mock chunk 流为空")
        return MockRecording(
            info=self.info,
            samples=np.concatenate([chunk.samples for chunk in chunks], axis=1),
            markers=self.markers,
        )


def _channel_names(channels: int) -> tuple[str, ...]:
    if channels < len(MODEL_CHANNELS):
        raise ValueError(f"channels 不能小于模型通道数 {len(MODEL_CHANNELS)}")
    names = list(MODEL_CHANNELS)
    names.extend(f"AUX{index:02d}" for index in range(1, channels - len(names) + 1))
    return tuple(names)


def _check_positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} 必须是有限正数，收到 {value!r}")
    return number


def make_mock_chunks(
    *,
    duration_sec: float = 24.0,
    trial_sec: float = 4.0,
    sample_rate_hz: float = 160.0,
    chunk_ms: float = 50.0,
    channels: int = 64,
    subject: str = "S999",
    recording_id: str = "S999R04",
    seed: int = 20261105,
    noise_uv: float = 8.0,
    class_effect: float = 0.35,
    artifact: str = "none",
) -> tuple[SourceInfo, tuple[ReferenceMarker, ...], list[EEGChunk]]:
    """生成一条 mock 记录及其 ``EEGChunk`` 块。

    信号：每个模型通道 10 Hz + 20 Hz 正弦，幅度约 6 µV；T1 左手想象让左侧
    通道幅度 +35%、右侧 -35%，T2 反过来；T0 两侧相同。其它通道只有噪声。
    ``artifact`` 可选 ``spike`` / ``flatline`` / ``nan``，用于测质量门。
    """
    duration = _check_positive(duration_sec, "duration_sec")
    trial = _check_positive(trial_sec, "trial_sec")
    rate = _check_positive(sample_rate_hz, "sample_rate_hz")
    chunk_length = _check_positive(chunk_ms, "chunk_ms")
    if artifact not in _ARTIFACTS:
        raise ValueError(f"artifact 必须是 {_ARTIFACTS} 之一，收到 {artifact!r}")
    if isinstance(channels, bool) or not isinstance(channels, int):
        raise ValueError("channels 必须是整数")

    chunk_samples = max(1, int(round(chunk_length / 1000.0 * rate)))
    n_samples = int(round(duration * rate))
    n_samples = (n_samples // chunk_samples) * chunk_samples
    if n_samples < chunk_samples:
        raise ValueError("duration 太短，至少需要一个完整块")

    channel_names = _channel_names(channels)
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, float(noise_uv) * 1e-6, (channels, n_samples))
    time = np.arange(n_samples) / rate

    trial_samples = max(chunk_samples, int(round(trial * rate)))
    markers: list[ReferenceMarker] = []
    for index, start in enumerate(range(0, n_samples, trial_samples)):
        stop = min(start + trial_samples, n_samples)
        label = _TRIAL_LABELS[index % len(_TRIAL_LABELS)]
        markers.append(
            ReferenceMarker(
                recording_id=recording_id,
                onset_sample=start,
                label=label,
                duration_samples=stop - start,
            )
        )
        if stop - start < chunk_samples:
            continue

        direction = 0.0
        if label == "T1":
            direction = 1.0
        elif label == "T2":
            direction = -1.0
        trial_time = time[start:stop] - time[start]
        for name in MODEL_CHANNELS:
            row = channel_names.index(name)
            if name in _LEFT:
                side = 1.0
            elif name in _RIGHT:
                side = -1.0
            else:
                side = 0.0
            gain = 1.0 + class_effect * side * direction
            phase = rng.uniform(0.0, 2.0 * np.pi)
            wave = np.sin(2.0 * np.pi * 10.0 * trial_time + phase)
            wave += 0.6 * np.sin(2.0 * np.pi * 20.0 * trial_time + 1.3 * phase)
            samples[row, start:stop] += 6e-6 * gain * wave

    info = SourceInfo(
        source_id=MOCK_SOURCE_ID,
        recording_id=recording_id,
        sample_rate_hz=rate,
        channel_names=channel_names,
        subject_id=subject,
        run="4",
        n_samples=n_samples,
    )

    if artifact == "spike":
        samples[0, n_samples // 2] += 800e-6
    elif artifact == "flatline":
        flat_start = n_samples // 3
        flat_stop = min(n_samples, flat_start + int(round(1.5 * rate)))
        samples[1, flat_start:flat_stop] = 0.0
    elif artifact == "nan":
        samples[2, n_samples // 2] = np.nan

    chunks: list[EEGChunk] = []
    for seq, start in enumerate(range(0, n_samples, chunk_samples)):
        stop = start + chunk_samples
        chunks.append(
            EEGChunk(
                source_id=MOCK_SOURCE_ID,
                recording_id=recording_id,
                seq=seq,
                start_sample=start,
                sample_rate_hz=rate,
                channel_names=channel_names,
                samples=samples[:, start:stop],
                released_monotonic_ms=seq * chunk_ms,
                discontinuity=(seq == 0),
            )
        )

    guard = ChunkStreamGuard(info)
    for chunk in chunks:
        guard.validate(chunk)
    return info, tuple(markers), chunks


def save_mock_chunks(
    path: str | Path,
    info: SourceInfo,
    markers: tuple[ReferenceMarker, ...],
    chunks: list[EEGChunk],
    *,
    meta: dict[str, object] | None = None,
) -> MockChunkStream:
    """把块流写成单个 ``.npz``；不进 pickle，数组键固定。"""
    target = Path(path)
    if target.suffix != ".npz":
        raise ValueError(f"mock 文件必须以 .npz 结尾，收到 {target.name}")
    if not chunks:
        raise ValueError("chunks 不能为空")
    chunk_samples = chunks[0].n_samples
    if any(chunk.n_samples != chunk_samples for chunk in chunks):
        raise ValueError("当前 mock 只支持固定块长")
    if any(chunk.channel_names != info.channel_names for chunk in chunks):
        raise ValueError("块的通道表必须与 info 一致")

    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": CONTRACT_VERSION,
        "source_id": info.source_id,
        "recording_id": info.recording_id,
        "subject_id": info.subject_id or "",
        "run": info.run or "",
        "sample_rate_hz": info.sample_rate_hz,
        "channel_names": np.asarray(info.channel_names),
        "channel_types": np.asarray(chunks[0].channel_types),
        "n_samples": info.n_samples,
        "n_chunks": len(chunks),
        "chunk_samples": chunk_samples,
        "seq": np.asarray([chunk.seq for chunk in chunks], dtype=np.int64),
        "start_sample": np.asarray([chunk.start_sample for chunk in chunks], dtype=np.int64),
        "released_monotonic_ms": np.asarray(
            [chunk.released_monotonic_ms for chunk in chunks], dtype=np.float64
        ),
        "discontinuity": np.asarray([chunk.discontinuity for chunk in chunks], dtype=bool),
        "samples": np.stack([chunk.samples for chunk in chunks], axis=0),
        "marker_onset_sample": np.asarray(
            [marker.onset_sample for marker in markers], dtype=np.int64
        ),
        "marker_duration_samples": np.asarray(
            [marker.duration_samples or 0 for marker in markers], dtype=np.int64
        ),
        "marker_label": np.asarray([marker.label for marker in markers]),
        "meta_json": np.asarray(
            json.dumps(
                {"created_at_utc": datetime.now(UTC).isoformat(), **(meta or {})},
                ensure_ascii=False,
            )
        ),
    }
    np.savez(target, **payload)
    return MockChunkStream(
        path=target,
        info=info,
        markers=markers,
        n_chunks=len(chunks),
        chunk_samples=chunk_samples,
    )


def create_mock_chunk_file(
    path: str | Path,
    *,
    duration_sec: float = 24.0,
    trial_sec: float = 4.0,
    sample_rate_hz: float = 160.0,
    chunk_ms: float = 50.0,
    channels: int = 64,
    subject: str = "S999",
    recording_id: str = "S999R04",
    seed: int = 20261105,
    noise_uv: float = 8.0,
    class_effect: float = 0.35,
    artifact: str = "none",
) -> MockChunkStream:
    """生成并落盘一条 mock chunk 流。"""
    info, markers, chunks = make_mock_chunks(
        duration_sec=duration_sec,
        trial_sec=trial_sec,
        sample_rate_hz=sample_rate_hz,
        chunk_ms=chunk_ms,
        channels=channels,
        subject=subject,
        recording_id=recording_id,
        seed=seed,
        noise_uv=noise_uv,
        class_effect=class_effect,
        artifact=artifact,
    )
    return save_mock_chunks(
        path,
        info,
        markers,
        chunks,
        meta={
            "duration_sec": duration_sec,
            "trial_sec": trial_sec,
            "chunk_ms": chunk_ms,
            "seed": seed,
            "noise_uv": noise_uv,
            "class_effect": class_effect,
            "artifact": artifact,
        },
    )


def _read_stream_data(path: str | Path) -> dict[str, np.ndarray]:
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"找不到 mock 文件：{target}")
    with np.load(target, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_mock_markers(path: str | Path) -> tuple[ReferenceMarker, ...]:
    """读 mock 的 ``ReferenceMarker``。"""
    data = _read_stream_data(path)
    recording_id = str(data["recording_id"].item())
    onsets = data["marker_onset_sample"].tolist()
    durations = data["marker_duration_samples"].tolist()
    labels = data["marker_label"].tolist()
    return tuple(
        ReferenceMarker(
            recording_id=recording_id,
            onset_sample=int(onset),
            label=str(label),
            duration_samples=int(duration),
        )
        for onset, duration, label in zip(onsets, durations, labels, strict=True)
    )


def load_mock_chunks(path: str | Path) -> Iterator[EEGChunk]:
    """逐块读 mock ``EEGChunk``；校验字段由 ``EEGChunk`` 构造器完成。"""
    data = _read_stream_data(path)
    samples = data["samples"]
    if samples.ndim != 3:
        raise ValueError(f"mock samples 必须是 (块, 通道, 样本)，收到 {samples.shape}")
    channel_names = tuple(str(name) for name in data["channel_names"].tolist())
    channel_types = tuple(str(kind) for kind in data["channel_types"].tolist())
    source_id = str(data["source_id"].item())
    recording_id = str(data["recording_id"].item())
    rate = float(data["sample_rate_hz"].item())
    for index in range(int(data["n_chunks"].item())):
        yield EEGChunk(
            source_id=source_id,
            recording_id=recording_id,
            seq=int(data["seq"][index]),
            start_sample=int(data["start_sample"][index]),
            sample_rate_hz=rate,
            channel_names=channel_names,
            channel_types=channel_types,
            samples=samples[index],
            released_monotonic_ms=float(data["released_monotonic_ms"][index]),
            discontinuity=bool(data["discontinuity"][index]),
        )


def load_mock_stream(path: str | Path) -> MockChunkStream:
    """读文件头并返回可迭代的 chunk 流。"""
    data = _read_stream_data(path)
    info = SourceInfo(
        source_id=str(data["source_id"].item()),
        recording_id=str(data["recording_id"].item()),
        sample_rate_hz=float(data["sample_rate_hz"].item()),
        channel_names=tuple(str(name) for name in data["channel_names"].tolist()),
        subject_id=str(data["subject_id"].item()) or None,
        run=str(data["run"].item()) or None,
        n_samples=int(data["n_samples"].item()),
    )
    return MockChunkStream(
        path=Path(path),
        info=info,
        markers=load_mock_markers(path),
        n_chunks=int(data["n_chunks"].item()),
        chunk_samples=int(data["chunk_samples"].item()),
    )
