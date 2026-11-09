"""A 的会话存档接线。生产循环和动作边界由上游提供。"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Event, Lock, Thread, get_ident
from typing import Literal

from eeg_core.contracts import (
    ContractError,
    DecisionFrame,
    IntentEvent,
    RuntimeEvent,
    RuntimeEventKind,
    SessionState,
)

from ..outputs import ArchiveWriteError, JsonlResultSink

type EndReason = Literal["finished", "stopped"]
type ArchiveItem = DecisionFrame | RuntimeEvent | IntentEvent

_IDENTITY_FIELDS = ("recording_id", "model_id", "pipeline_version", "config_hash")
_ERROR_POLL_SEC = 0.05


class ArchivedSession:
    """一个工作线程负责生产、入队和收尾；D 的线程负责写盘。

    ``produce`` 在同一个工作线程调用 ``submit``，退出时返回 EOF 或用户停止。
    ``stop_production`` 必须线程安全、幂等，并能唤醒阻塞供数（如 ReplaySource.stop）。
    界面回调只投递排队信号。错误回调也可能由存档状态检查线程发出，
    因此回调不能等待会话工作线程退出。
    """

    def __init__(
        self,
        output_root: str | Path,
        session_id: str,
        *,
        stop_production: Callable[[], None],
        emit_decision: Callable[[DecisionFrame], None],
        emit_archive_error: Callable[[str, str], None],
    ) -> None:
        self._output_root = output_root
        self._session_id = session_id
        self._stop_production = stop_production
        self._emit_decision = emit_decision
        self._emit_archive_error = emit_archive_error
        self._lock = Lock()
        self._sink: JsonlResultSink | None = None
        self._archive_error: str | None = None
        self._worker_ident: int | None = None
        self._accepting = False
        self._identity: dict[str, object] = {}
        self._next_seq = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def status(self) -> dict[str, object]:
        """供 GUI 定时读取的保存状态快照，不等待磁盘或工作线程。"""
        with self._lock:
            sink, error = self._sink, self._archive_error
        status = sink.status if sink is not None else {
            "state": "opening",
            "directory": None,
            "error": None,
            "queued": 0,
            "completed": False,
        }
        if error is not None:
            status.update(state="failed", error=error, completed=False)
        return status

    def submit(self, item: ArchiveItem) -> None:
        """同一工作线程按生产顺序提交，DecisionFrame 入队后原样发给界面。"""
        if self._worker_ident != get_ident() or not self._accepting or self._sink is None:
            raise RuntimeError("只允许当前会话的生产循环提交结果")
        self._enqueue(self._sink, item)

    def _enqueue(self, sink: JsonlResultSink, item: ArchiveItem) -> None:
        if item.session_id != self.session_id:
            raise ContractError("结果不属于当前会话")
        if isinstance(item, DecisionFrame):
            # 在投递 UI 信号前核对身份和序号，避免等到 D 异步拒绝时才发现。
            if item.seq != self._next_seq:
                raise ContractError(f"期望决策 seq={self._next_seq}，收到 {item.seq}")
            for field, expected in self._identity.items():
                if getattr(item, field) != expected:
                    raise ContractError(f"决策的 {field} 与本会话不一致")
        elif isinstance(item, IntentEvent):
            if item.end_decision_seq is None or item.end_decision_seq >= self._next_seq:
                raise ContractError("动作事件必须已结束，且先提交它引用的决策")
        sink.submit(item)
        if isinstance(item, DecisionFrame):
            self._next_seq += 1
            self._emit_decision(item)

    def run(
        self,
        metadata: Mapping[str, object],
        produce: Callable[[ArchivedSession], EndReason],
        close_intent: Callable[[EndReason], IntentEvent | None],
    ) -> SessionState:
        """在控制工作线程打开存档、运行生产循环、收尾并等待 D 写完。

        ``close_intent`` 使用上游已确定的动作边界和决策序号。只有当前没有
        待关闭动作时才返回 None。读取结束和用户停止均可产生完整存档。
        """
        with self._lock:
            if self._worker_ident is not None:
                raise RuntimeError("会话已启动。重新开始时必须创建新会话")
            self._worker_ident = get_ident()

        sink = None
        watcher = None
        stop_watching = Event()
        close_attempted = False
        try:
            sink = JsonlResultSink(self._output_root, self.session_id, metadata)
            with self._lock:
                self._sink = sink
            self._identity = {field: metadata[field] for field in _IDENTITY_FIELDS}
            sink.submit(self._runtime_event(RuntimeEventKind.SESSION_START))
            watcher = Thread(
                target=self._watch_archive,
                args=(sink, stop_watching),
                name=f"archive-status-{self.session_id}",
                daemon=True,
            )
            watcher.start()

            self._accepting = True
            try:
                reason = produce(self)
            finally:
                self._accepting = False
                self._stop_production()

            sink.check_error()
            if reason not in ("finished", "stopped"):
                raise ValueError("生产循环必须报告 finished 或 stopped")
            if (event := close_intent(reason)) is not None:
                if event.end_reason != reason:
                    raise ContractError("当前动作的结束原因应与会话结束原因一致")
                self._enqueue(sink, event)
            kind = RuntimeEventKind.FINISH if reason == "finished" else RuntimeEventKind.STOP
            sink.submit(self._runtime_event(kind))
            close_attempted = True
            sink.close(completed=True, reason=reason)
            return SessionState.FINISHED if reason == "finished" else SessionState.STOPPED
        except ArchiveWriteError as error:
            self._fail_archive(error)
            return SessionState.ERROR
        finally:
            stop_watching.set()
            if watcher is not None and watcher.ident is not None:
                watcher.join()
            if sink is not None and not close_attempted:
                abort_reason = "write_error" if self._archive_error is not None else "runtime_error"
                try:
                    sink.close(completed=False, reason=abort_reason)
                except ArchiveWriteError as error:
                    self._fail_archive(error)

    def _runtime_event(self, kind: RuntimeEventKind) -> RuntimeEvent:
        return RuntimeEvent(self.session_id, time.monotonic() * 1000, kind)

    def _watch_archive(self, sink: JsonlResultSink, stop_watching: Event) -> None:
        # 即使生产在暂停或等待输入，也要检查后台 IO 是否失败。
        while not stop_watching.wait(_ERROR_POLL_SEC):
            try:
                sink.check_error()
            except ArchiveWriteError as error:
                self._fail_archive(error)
                return

    def _fail_archive(self, error: ArchiveWriteError) -> None:
        with self._lock:
            if self._archive_error is not None:
                return
            self._archive_error = str(error)
        try:
            self._stop_production()
        finally:
            self._emit_archive_error(self.session_id, str(error))
