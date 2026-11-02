"""界面测试回放控制器（独立线程）—— C 的 UI 测试驱动。

职责（C 的分工 2.3）：
- 在独立线程中驱动数据源与解码器，保证耗时操作不在主线程执行，
  不阻塞解码与界面（PRD 6.5：主线程只负责交互与绘制）。
- 同时消费两条接口：数据源（结构约定 ``SourceProtocol``，来自 A）与解码器（算法，来自 B）。
  二者都可注入，因此"接入设备"与"匹配 B 的算法"对 C 的界面完全透明。
- 支持 开始 / 暂停 / 继续 / 停止；暂停时回放源时钟停止（PRD 6.4）。
- 以 ~50 ms 粒度释放 ``EEGChunk``；解码器自行决定 ``DecisionFrame`` 的发布节奏（PRD 6.1）。

注意：本模块是**界面测试**阶段对 A 的运行时（``eeg_app/runtime``）的临时替代。
正式集成时由 A 的运行时会话控制器驱动，本文件将被移除或仅作为测试夹具保留。
解码器的调用约定（``set_session`` / ``reset(reason)`` / ``decode(chunk)``）目前不在
团队契约的 ``StreamConsumer`` 协议内（契约仅定义了 ``reset(reason)``），待契约冻结后
需与 B 的运行时调用方式对齐——此处按界面测试最小可用约定实现。
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from PySide6.QtCore import QThread, Signal, QElapsedTimer

from eeg_core.contracts import (
    EEGChunk,
    DecisionFrame,
    ResetReason,
    SourceInfo,
)

CHUNK_MS = 50        # ~8 样本 @160 Hz（PRD 6.1 目标 50 ms 一块）


@runtime_checkable
class SourceProtocol(Protocol):
    """界面测试用数据源的结构约定（留后手 / 防御性契约）。

    B 的冻结契约 v1 把 ``DataSource`` 从 ``eeg_core.contracts`` 中移除，
    因此 C 的 UI 不再直接依赖该符号，只依赖它真正用到的"形状"：

    * ``info``：``SourceInfo``，首次读取即可得到记录元数据；
    * ``read_chunk()``：返回下一块 ``EEGChunk``，有限源结束后返回 ``None``；
    * 支持上下文管理器（``with source:``）以释放资源。

    任何满足上述结构的对象（EDF 回放 / 真实设备 / 模拟源）都可注入，
    接入 A 的真实数据源时 C 的界面与回放控制器无需改动。
    """

    info: SourceInfo

    def read_chunk(self) -> Optional[EEGChunk]:
        ...

    def __enter__(self) -> "SourceProtocol":
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        ...


@runtime_checkable
class DecoderProtocol(Protocol):
    """界面测试用解码器最小约定（待契约冻结后与 B 对齐）。

    解码器是状态ful 的流式消费者，除 ``StreamConsumer.reset(reason)`` 外，
    还需 ``set_session`` 与 ``decode`` 供运行时/回放控制器调用。
    """

    def set_session(self, session_id: str) -> None:
        """新会话开始时写入会话标识（用于 DecisionFrame.session_id）。"""
        ...

    def reset(self, reason: ResetReason) -> None:
        """按原因清理内部状态（见 ``StreamConsumer`` 语义）。"""
        ...

    def decode(self, chunk: EEGChunk) -> Optional[DecisionFrame]:
        """输入一个数据块，返回该块对应的决策；若尚未达到发布条件返回 None。"""
        ...


class PlaybackWorker(QThread):
    chunk_ready = Signal(EEGChunk)
    decision_ready = Signal(DecisionFrame)
    playback_started = Signal()
    playback_paused = Signal()
    playback_resumed = Signal()
    playback_stopped = Signal()
    playback_finished = Signal()

    def __init__(
        self,
        parent=None,
        source: Optional[SourceProtocol] = None,
        decoder: Optional[DecoderProtocol] = None,
    ) -> None:
        super().__init__(parent)
        # 任何实现 SourceProtocol 结构的对象都可注入（EDF 回放 / 真实设备 / 未来设备）。
        if source is None:
            raise ValueError("PlaybackWorker 需要一个数据源（实现 SourceProtocol）")
        self.source = source
        self.sample_rate_hz = source.info.sample_rate_hz
        # 任何实现 DecoderProtocol 的对象都可注入（B 的连续解码算法）。
        self.decoder = decoder
        self.session_id = ""
        self._running = False
        self._paused = False
        self._suppress_decisions = False  # N08 演示：暂停决策发布，但数据继续流式供数

    def set_session(self, session_id: str) -> None:
        self.session_id = session_id

    def run(self) -> None:
        # 新会话：先给解码器写入会话标识并清空状态
        if self.decoder is not None:
            if hasattr(self.decoder, "set_session"):
                self.decoder.set_session(self.session_id)
            self.decoder.reset(ResetReason.SESSION_START)
        self._running = True
        self._paused = False
        clock = QElapsedTimer()
        clock.start()
        self.playback_started.emit()

        with self.source:
            while self._running:
                if self._paused:
                    self.msleep(20)
                    continue
                mono = clock.elapsed()           # 会话单调时钟（用于运行时延测量）
                chunk = self.source.read_chunk()
                if chunk is None:
                    # 有限源（如文件）结束：自然完成会话
                    self.playback_finished.emit()
                    break
                self.chunk_ready.emit(chunk)
                # 解码器决定何时发布决策（B 的真实算法即如此）
                # N08 演示：若开启"决策中断"，则只断决策、不断数据（波形继续滚动）
                if self.decoder is not None and not self._suppress_decisions:
                    try:
                        df = self.decoder.decode(chunk)
                    except Exception as exc:  # 单帧异常不应中断整个回放
                        print(f"[PlaybackWorker] 解码异常（已跳过该块）: {exc!r}")
                    else:
                        if df is not None:
                            self.decision_ready.emit(df)
                self.msleep(CHUNK_MS)

        self.playback_stopped.emit()

    def pause(self) -> None:
        if self._running and not self._paused:
            self._paused = True
            self.playback_paused.emit()

    def resume(self) -> None:
        if self._running and self._paused:
            self._paused = False
            # 继续运行：作废暂停前的候选确认计数（PAUSE_RESUME 语义），但不丢历史
            if self.decoder is not None:
                self.decoder.reset(ResetReason.PAUSE_RESUME)
            self.playback_resumed.emit()

    def stop(self) -> None:
        self._running = False

    def set_suppress_decisions(self, on: bool) -> None:
        """N08 演示：暂停/恢复决策发布（数据仍继续流式供数）。"""
        self._suppress_decisions = bool(on)
