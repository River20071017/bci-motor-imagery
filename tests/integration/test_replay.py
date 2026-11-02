"""集成：记录 → :class:`ReplaySource` → :class:`ChunkStreamGuard`（T03 / T05 / T06 / T11）。

只依赖 numpy 与契约：合成记录就能覆盖分块、排程、暂停/停止，不要求本机有 EDF。
最后一条用例才读真实 EDF，文件或 mne 缺失时跳过。

没有 pytest 也能跑：``python tests/integration/test_replay.py``（需要 ``src`` 在 PYTHONPATH 上，
或已 ``pip install -e .``）。
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from eeg_core.contracts import (
    DEFAULT_CONFIG,
    ChunkStreamGuard,
    ContractError,
    IncompatibleSourceError,
    SourceInfo,
)
from eeg_app.sources import (
    Recording,
    ReplaySource,
    SourceState,
    chunk_samples_for,
    standardize_channel_names,
)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "eegmmidb"
MODEL_CHANNELS = DEFAULT_CONFIG.source.model_channels


# --------------------------------------------------------------------- 工具


def _channel_names(count: int = 64) -> tuple[str, ...]:
    """前 9 个是模型通道，其余补齐到 64 个不重名通道。"""
    if count < len(MODEL_CHANNELS):
        return tuple(MODEL_CHANNELS[:count])
    fillers = tuple(f"E{index:02d}" for index in range(1, count - len(MODEL_CHANNELS) + 1))
    return tuple(MODEL_CHANNELS) + fillers


def _recording(
    seconds: float = 1.0,
    *,
    sample_rate_hz: float = 160.0,
    channels: int = 64,
    record_id: str = "S001R04",
) -> Recording:
    """一条可控的合成记录：第 c 通道第 t 个样本恒为 ``c * 1e-6 + t * 1e-9``，便于核对切片。"""
    n_samples = int(round(seconds * sample_rate_hz))
    names = _channel_names(channels)
    data = (
        np.arange(channels, dtype=np.float64)[:, None] * 1e-6
        + np.arange(n_samples, dtype=np.float64)[None, :] * 1e-9
    )
    info = SourceInfo(
        source_id="synthetic",
        recording_id=record_id,
        sample_rate_hz=sample_rate_hz,
        channel_names=names,
        n_samples=n_samples,
    )
    return Recording(info=info, samples=data)


def _fast(speed: float = 500.0):
    """把回放倍速调到很高，让用例不用真等 125 秒。"""
    return replace(
        DEFAULT_CONFIG,
        source=replace(DEFAULT_CONFIG.source, replay_speed=speed),
    )


def _skip(reason: str) -> None:
    try:
        import pytest  # noqa: PLC0415
    except ImportError:
        print(f"SKIP: {reason}")
        return
    pytest.skip(reason)


def _read_all(source: ReplaySource, limit: int | None = None) -> list:
    chunks = []
    guard = ChunkStreamGuard(source.info)
    while (chunk := source.read_chunk()) is not None:
        guard.validate(chunk)
        chunks.append(chunk)
        if limit is not None and len(chunks) >= limit:
            break
    return chunks


# --------------------------------------------------------------------- 用例


def test_standardize_channel_names() -> None:
    """PhysioNet 的 ``Fc5.`` 要变成模型认得的 ``FC5``；重名必须报错而不是猜。"""
    assert standardize_channel_names(["Fc5.", "Fc3.", "Fcz.", "Cz.", "Fp1.", "Cp3."]) == (
        "FC5",
        "FC3",
        "FCz",
        "Cz",
        "Fp1",
        "CP3",
    )
    assert standardize_channel_names([" FC3. "]) == ("FC3",)

    for bad in (["Cz.", "CZ."], ["C3.", "c3."], ["  "], []):
        try:
            standardize_channel_names(bad)
        except ContractError:
            continue
        raise AssertionError(f"应当拒绝：{bad}")


def test_chunk_stream_passes_guard() -> None:
    """首块标间断、seq 连续、首尾相接，全部交给 ChunkStreamGuard 判定。"""
    source = ReplaySource(_recording(seconds=1.0), config=_fast())
    chunks = _read_all(source)

    assert source.chunk_samples == chunk_samples_for(DEFAULT_CONFIG) == 8
    assert len(chunks) == 160 // 8  # 1 秒 = 160 个样本 = 20 块
    assert chunks[0].seq == 0 and chunks[0].discontinuity is True
    assert all(not chunk.discontinuity for chunk in chunks[1:])
    assert chunks[0].unit == "V"
    assert source.state is SourceState.FINISHED
    assert source.read_chunk() is None


def test_block_length_is_free() -> None:
    """块长换一下也要过（T05）：校验器只认连续性，不认块长。"""
    for size in (1, 8, 13, 160):
        source = ReplaySource(_recording(seconds=0.5), config=_fast(), chunk_samples=size)
        chunks = _read_all(source)
        assert len(chunks) == -(-source.n_samples // size)
        assert all(chunk.n_samples == size for chunk in chunks[:-1])


def test_coverage_and_partial_last_chunk() -> None:
    """样本不重不漏；最后一块可以短，但必须落在记录末尾。"""
    recording = _recording(seconds=0.5)  # 80 个样本
    source = ReplaySource(recording, config=_fast(), chunk_samples=13)
    chunks = _read_all(source)

    assert sum(chunk.n_samples for chunk in chunks) == recording.n_samples
    assert chunks[-1].end_sample == recording.n_samples
    assert chunks[-1].n_samples == 80 % 13
    # 切片对得上：第 0 通道第 t 个样本 = t * 1e-9
    assert np.allclose(chunks[0].samples[0], np.arange(0, 13) * 1e-9)
    assert np.allclose(chunks[1].samples[0], np.arange(13, 26) * 1e-9)


def test_markers_never_enter_chunks() -> None:
    """标签隔离：块里只有 ``discontinuity``，参考标注只在 ``markers`` 上。"""
    source = ReplaySource(_recording(seconds=0.2), config=_fast())
    chunk = source.read_chunk()
    assert chunk is not None
    assert "T0" not in repr(chunk)
    assert set(chunk.metadata()) == {
        "schema_version",
        "source_id",
        "recording_id",
        "seq",
        "start_sample",
        "sample_rate_hz",
        "channel_names",
        "channel_types",
        "released_monotonic_ms",
        "discontinuity",
    }
    assert hasattr(source, "markers")
    source.stop()


def test_released_time_is_monotonic() -> None:
    """``released_monotonic_ms`` 只增不减，且都 ≥ 0。"""
    source = ReplaySource(_recording(seconds=0.3), config=_fast())
    released = [chunk.released_monotonic_ms for chunk in _read_all(source)]
    assert released == sorted(released)
    assert all(value >= 0 for value in released)


def test_stop_is_immediate() -> None:
    """停止后阻塞中的 ``read_chunk`` 立刻返回 ``None``（N07 的 250 ms 靠这个）。"""
    source = ReplaySource(_recording(seconds=10.0), config=_fast(speed=1.0))
    assert source.read_chunk() is not None  # 先吃掉首块，下一次会等到 50 ms 才释放

    result: list = []

    def worker() -> None:
        result.append(source.read_chunk())

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.02)
    started = time.perf_counter()
    source.stop()
    thread.join(timeout=1.0)
    stopped_in = time.perf_counter() - started

    assert not thread.is_alive()
    assert result == [None]
    assert stopped_in < 0.25, stopped_in
    assert source.state is SourceState.STOPPED
    assert source.read_chunk() is None


def test_pause_blocks_and_resume_keeps_sequence() -> None:
    """暂停期间不产块，恢复后接着上一块，不跳样本、不补吐。"""
    source = ReplaySource(_recording(seconds=10.0), config=_fast(speed=1.0))
    assert source.read_chunk() is not None

    result: list = []

    def worker() -> None:
        result.append(source.read_chunk())

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.01)
    assert source.pause() is True
    assert source.state is SourceState.PAUSED
    time.sleep(0.15)
    assert thread.is_alive(), "暂停期间 read_chunk 必须一直等"
    assert result == []

    assert source.resume() is True
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    (chunk,) = result
    assert chunk.seq == 1
    assert chunk.start_sample == source.chunk_samples
    source.stop()


def test_realtime_pacing_follows_source_time() -> None:
    """1 倍速下 0.5 秒数据要花约 0.5 秒放完（首块立即释放）。"""
    source = ReplaySource(_recording(seconds=0.5), config=_fast(speed=1.0), chunk_samples=40)
    started = time.perf_counter()
    chunks = _read_all(source)
    elapsed = time.perf_counter() - started

    assert len(chunks) == 2
    assert 0.2 <= elapsed <= 1.2, elapsed


def test_incompatible_record_is_rejected() -> None:
    """记录缺模型通道、或采样率不符时一次报全原因（T02），不进入回放。"""
    names = tuple(f"X{index:02d}" for index in range(64))
    info = SourceInfo(
        source_id="synthetic",
        recording_id="S099R04",
        sample_rate_hz=160.0,
        channel_names=names,
        n_samples=160,
    )
    recording = Recording(info=info, samples=np.zeros((64, 160)))
    try:
        ReplaySource(recording, config=DEFAULT_CONFIG)
    except IncompatibleSourceError as error:
        assert "缺少模型必需通道" in str(error)
        assert len(error.reasons) >= 1
    else:
        raise AssertionError("缺模型通道的记录不应通过")

    healthy = _recording(seconds=1.0)
    wrong_rate = Recording(
        info=replace(healthy.info, sample_rate_hz=250.0),
        samples=healthy.samples,
    )
    try:
        ReplaySource(wrong_rate, config=DEFAULT_CONFIG)
    except IncompatibleSourceError as error:
        assert "采样率不符" in str(error)
    else:
        raise AssertionError("250 Hz 的记录不应按 160 Hz 跑")


def test_real_edf_replay() -> None:
    """真实 EDF：S001R04 读出来是 64 通道 / 160 Hz / 125 s，前 100 块过校验。"""
    path = DATA_ROOT / "S001" / "S001R04.edf"
    if not path.is_file():
        _skip(f"没有样例数据：{path}")
        return

    from eeg_app.sources import load_recording  # noqa: PLC0415 - 延迟导入，缺 mne 时才跳过

    try:
        recording = load_recording(DATA_ROOT, "S001R04")
    except ContractError as error:  # pragma: no cover - 依赖缺失或文件损坏
        _skip(f"读不了 EDF：{error}")
        return

    assert len(recording.info.channel_names) == 64
    assert recording.info.sample_rate_hz == 160.0
    assert recording.info.n_samples == 20000
    assert all(name in recording.info.channel_names for name in MODEL_CHANNELS)
    assert recording.info.reference is None
    assert recording.info.reference_label == "未提供"
    assert recording.sha256 and len(recording.sha256) == 64
    assert set(recording.labels) <= {"T0", "T1", "T2"}

    source = ReplaySource(recording, config=_fast())
    chunks = _read_all(source, limit=100)
    assert [chunk.seq for chunk in chunks] == list(range(100))
    assert chunks[-1].start_sample == 99 * source.chunk_samples
    source.stop()


def _run_all() -> int:
    names = [name for name in sorted(globals()) if name.startswith("test_")]
    failures = 0
    for name in names:
        function = globals()[name]
        started = time.perf_counter()
        try:
            function()
        except AssertionError as error:
            failures += 1
            print(f"FAIL {name}: {error}")
        except Exception as error:  # noqa: BLE001 - 自检脚本要打印全部失败
            failures += 1
            print(f"ERROR {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {name} ({time.perf_counter() - started:.2f}s)")
    print(f"\n{len(names) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
