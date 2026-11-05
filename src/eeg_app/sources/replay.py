"""1 倍速回放：把一条记录按原始采样时序切成块，交给 runtime。

本模块只管「什么时候给下一块」，不管样本怎么读（:mod:`eeg_app.sources.input` 负责）。
契约规则（docs/01 §3）全部在这里兑现：

* 首块 ``seq=0`` 且 ``discontinuity=True``，之后 ``seq`` 严格 +1；
* 未标间断的块 ``start_sample`` 必须等于上一块 ``end_sample``；
* 块长只由 ``SourceConfig.target_chunk_ms`` 决定（目标采样率 × 50 ms），
  校验器不检查块长，块长换一下也得过（T05）；
* ``released_monotonic_ms`` 记这一块**真正交出去**的会话单调时钟，只用于算延迟。

用法：

.. code-block:: python

    from eeg_core.contracts import DEFAULT_CONFIG, ChunkStreamGuard
    from eeg_app.sources import ReplaySource, load_recording

    source = ReplaySource(load_recording("data/eegmmidb", "S001R04"), config=DEFAULT_CONFIG)
    guard = ChunkStreamGuard(source.info)
    source.start()
    while (chunk := source.read_chunk()) is not None:
        guard.validate(chunk)

节奏三条约定（F02 / T03 / T06）：

1. **按源时间排程**：第 i 块（起始样本 ``s``）的释放时刻是
   ``起点 + s / (采样率 × 回放倍速)``，与「读了多少块」无关；处理慢导致的落后不会被
   跳过——数据一块不少，积压交给 runtime 的环形缓冲与队列去判断过载。
2. **暂停不推进源时间**：暂停期间不产生块、不丢样本；恢复后暂停的那段时间整体后移，
   不会出现「一恢复就狂吐几十块」。
3. **停止立即生效**：``stop()`` 会让阻塞在 ``read_chunk()`` 里的线程马上返回 ``None``，
   N07 的 250 ms 停止要求靠这一点实现。
"""

from __future__ import annotations

import threading
import time
from enum import StrEnum

from eeg_core.contracts import (
    DEFAULT_CONFIG,
    AppConfig,
    EEGChunk,
    SourceInfo,
)

from .input import Recording, load_edf

__all__ = ["SourceState", "ReplaySource", "chunk_samples_for", "open_replay"]


class SourceState(StrEnum):
    """回放状态；与 README §4.3 的会话状态机对齐（``Error`` 不在这里，抛异常即失败）。"""

    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    STOPPED = "stopped"


_TERMINAL = frozenset({SourceState.FINISHED, SourceState.STOPPED})
"""不可再产出样本的状态。"""

_MAX_SLEEP_SEC = 0.005
"""单次睡眠上限。

Windows 上 ``Condition.wait()`` / ``Event.wait()`` 的超时会被 15.6 ms 系统节拍量化，
50 ms 一块会稳定迟到约 12 ms；``time.sleep()`` 在 Python 3.11+ 用的是高精度可等待定时器，
所以排程睡眠放在锁外用小步 ``time.sleep``，每步重新核对剩余时间（实测迟到 < 1 ms）。
"""


def chunk_samples_for(config: AppConfig) -> int:
    """按 ``target_chunk_ms`` 算每块样本数：目标采样率 × 50 ms，至少 1。"""
    return max(1, int(round(config.source.target_chunk_ms / 1000.0 * config.source.sample_rate_hz)))


class ReplaySource:
    """一条记录 + 一份配置 = 一个能按 1 倍速供数的数据源。

    构造时做完全部兼容性检查（T02）：缺模型通道、采样率不符都抛
    :class:`IncompatibleSourceError` 并一次列出全部原因，不带着不兼容的记录往下跑。
    通道数不检查：记录有多少通道是数据源自己的事（契约 1.1），算法只关心模型通道。
    参考方式（``reference_policy``）不在这里处理——那是处理层的重参考，不是文件头里的说法。
    """

    def __init__(
        self,
        recording: Recording,
        *,
        config: AppConfig = DEFAULT_CONFIG,
        chunk_samples: int | None = None,
        clock=time.monotonic,
    ) -> None:
        info = recording.info
        info.check_compatible(
            required_channels=config.source.model_channels,
            sample_rate_hz=config.source.sample_rate_hz,
        )

        samples = recording.samples
        size = chunk_samples if chunk_samples is not None else chunk_samples_for(config)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"chunk_samples 必须是大于 0 的整数，收到 {size!r}")

        self._recording = recording
        self._info: SourceInfo = info
        self._samples = samples
        self._n_samples = int(samples.shape[1])
        self._sample_rate_hz = info.sample_rate_hz
        self._speed = config.source.replay_speed
        self._chunk_samples = size
        self._clock = clock

        self._cond = threading.Condition()
        self._state = SourceState.READY
        self._cursor = 0
        self._seq = 0
        self._origin: float | None = None
        self._paused_total = 0.0
        self._paused_at: float | None = None

    # ---------------------------------------------------------------- 状态

    @property
    def state(self) -> SourceState:
        """当前回放状态。"""
        return self._state

    @property
    def info(self) -> SourceInfo:
        """记录档案卡；块里的 ``recording_id``、采样率、通道表都以它为准。"""
        return self._info

    @property
    def markers(self) -> tuple:
        """参考标注，单独一条线；**不要**把它们并进块或解码路径。"""
        return self._recording.markers

    @property
    def chunk_samples(self) -> int:
        """每块样本数。"""
        return self._chunk_samples

    @property
    def n_samples(self) -> int:
        """记录总样本数。"""
        return self._n_samples

    @property
    def duration_sec(self) -> float:
        """记录总时长（秒），由样本数推出。"""
        return self._n_samples / self._sample_rate_hz

    @property
    def position_samples(self) -> int:
        """下一块的首样本编号，也就是「源时间走到了哪」。"""
        return self._cursor

    @property
    def source_t_sec(self) -> float:
        """已消耗的源时间（秒），与块的 ``start_t_sec`` 同口径。"""
        return self._cursor / self._sample_rate_hz

    @property
    def progress(self) -> float:
        """回放进度 0~1。"""
        return self._cursor / self._n_samples if self._n_samples else 1.0

    @property
    def sent_chunks(self) -> int:
        """已经交出去的块数，等于下一块的 ``seq``。"""
        return self._seq

    # ---------------------------------------------------------------- 控制

    def start(self) -> None:
        """进入 ``running`` 并记下排程起点；重复调用无副作用，结束后再调抛 ``RuntimeError``。"""
        with self._cond:
            if self._state in _TERMINAL:
                raise RuntimeError(f"回放已结束（{self._state}），请新建 ReplaySource")
            if self._state in (SourceState.RUNNING, SourceState.PAUSED):
                return
            self._begin_locked()
            self._cond.notify_all()

    def pause(self) -> bool:
        """暂停供数；只有 ``running`` 时有效。返回是否真的暂停了。"""
        with self._cond:
            if self._state is not SourceState.RUNNING:
                return False
            self._state = SourceState.PAUSED
            self._paused_at = self._clock()
            self._cond.notify_all()
            return True

    def resume(self) -> bool:
        """继续供数；把暂停的时长整体加到排程上，不会补吐暂停期间的块。"""
        with self._cond:
            if self._state is not SourceState.PAUSED:
                return False
            if self._paused_at is not None:
                self._paused_total += self._clock() - self._paused_at
                self._paused_at = None
            self._state = (
                SourceState.RUNNING if self._origin is not None else SourceState.READY
            )
            self._cond.notify_all()
            return True

    def stop(self) -> None:
        """停止回放；阻塞在 ``read_chunk()`` 里的调用立即返回 ``None``。"""
        with self._cond:
            if self._state in _TERMINAL:
                return
            self._state = SourceState.STOPPED
            self._cond.notify_all()

    def close(self) -> None:
        """``stop`` 的别名，供 ``with`` 使用。"""
        self.stop()

    def __enter__(self) -> ReplaySource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ---------------------------------------------------------------- 供数

    def read_chunk(self) -> EEGChunk | None:
        """取下一块；到达文件末尾或已停止时返回 ``None``。

        会阻塞到这一块的释放时刻（1 倍速时大约 50 ms 一块）；暂停期间一直等，
        停止时在一步睡眠（≤ 5 ms）内返回 ``None``。首次调用会自动 ``start``。
        """
        while True:
            with self._cond:
                if self._state in _TERMINAL:
                    return None
                if self._state is SourceState.READY:
                    self._begin_locked()
                if self._state is SourceState.PAUSED:
                    self._cond.wait()
                    continue
                remaining = self._due_time_locked(self._cursor) - self._clock()
                if remaining <= 0.0:
                    return self._emit_locked()
            # 锁外睡：持锁睡会挡住 pause()/stop()，而 cond.wait() 在 Windows 上不准。
            time.sleep(min(remaining, _MAX_SLEEP_SEC))

    # ---------------------------------------------------------------- 内部

    def _begin_locked(self) -> None:
        self._state = SourceState.RUNNING
        self._origin = self._clock()
        self._paused_total = 0.0
        self._paused_at = None

    def _due_time_locked(self, start_sample: int) -> float:
        """起始样本 ``start_sample`` 这一块的应释放时刻（单调时钟，秒）。"""
        origin = self._origin if self._origin is not None else self._clock()
        source_seconds = start_sample / (self._sample_rate_hz * self._speed)
        return origin + self._paused_total + source_seconds

    def _emit_locked(self) -> EEGChunk:
        start = self._cursor
        stop = min(start + self._chunk_samples, self._n_samples)
        chunk = EEGChunk(
            source_id=self._info.source_id,
            recording_id=self._info.recording_id,
            seq=self._seq,
            start_sample=start,
            sample_rate_hz=self._sample_rate_hz,
            channel_names=self._info.channel_names,
            samples=self._samples[:, start:stop],
            released_monotonic_ms=self._clock() * 1000.0,
            discontinuity=self._seq == 0,
        )
        self._cursor = stop
        self._seq += 1
        if stop >= self._n_samples:
            self._state = SourceState.FINISHED
            self._cond.notify_all()
        return chunk


def open_replay(
    path: str,
    *,
    config: AppConfig = DEFAULT_CONFIG,
    chunk_samples: int | None = None,
) -> ReplaySource:
    """从 EDF 路径直接建回放源：读文件 + 兼容性检查一步到位。"""
    return ReplaySource(
        load_edf(path), config=config, chunk_samples=chunk_samples
    )
